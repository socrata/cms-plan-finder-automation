#!/usr/bin/env python3

"""A script to automate updates to CMS Plan Finder data on Socrata.

Requires a number of environment variables to be set prior to script
execution. See the README.md file for more information.
"""

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from contextlib import contextmanager
import csv
import json
import logging
from operator import itemgetter
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pendulum
from pendulum import Date, DateTime, Duration
import requests
from requests import Response
from socrata import Socrata
from socrata.authorization import Authorization
from socrata.job import Job
from socrata.output_schema import OutputSchema
from socrata.revisions import Revision
from socrata.sources import Source
from socrata.views import View
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Literal

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
file_formatter = logging.Formatter("[%(asctime)s] %(message)s")
file_handler = logging.FileHandler("socrata.log")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Constants
ENVS = ["test", "impl", "prod"]
TOKEN_URLS = {
    "test": "https://hpmstest.cms.gov/api/idm/OAuth/token",
    "impl": "https://hpmsimpl.cms.gov/api/idm/OAuth/token",
    "prod": "https://hpms.cms.gov/api/idm/OAuth/token",
}
USERNAMES = {
    "test": os.environ["SOCRATA_PLAN_FINDER_TEST_USERNAME"],
    "impl": os.environ["SOCRATA_PLAN_FINDER_IMPL_USERNAME"],
    "prod": os.environ["SOCRATA_PLAN_FINDER_PROD_USERNAME"],
}
DATA_URL = "https://hpms.cms.gov/api/mpfpe/planBenefits/downloadMctFiles"
API_KEYS = {
    "test": (
        os.environ["SOCRATA_PLAN_FINDER_TEST_KEY_ID"],
        os.environ["SOCRATA_PLAN_FINDER_TEST_KEY_SECRET"],
    ),
    "impl": (
        os.environ["SOCRATA_PLAN_FINDER_IMPL_KEY_ID"],
        os.environ["SOCRATA_PLAN_FINDER_IMPL_KEY_SECRET"],
    ),
    "prod": (
        os.environ["SOCRATA_PLAN_FINDER_PROD_KEY_ID"],
        os.environ["SOCRATA_PLAN_FINDER_PROD_KEY_SECRET"],
    ),
}
ACS_PARAMS = {
    "test": os.environ["SOCRATA_PLAN_FINDER_TEST_ACS"],
    "impl": os.environ["SOCRATA_PLAN_FINDER_IMPL_ACS"],
}
SOCRATA_DOMAIN = "plan-finder.data.medicare.gov"
SOCRATA_CREDENTIALS = (
    os.environ["SOCRATA_KEY_ID"],
    os.environ["SOCRATA_KEY_SECRET"],
)
SOCRATA_DATASET_PERMISSION = "public"  # Must be either "public" or "private"
FILES_TABLE_PATH = Path("./tables/files.csv")
SCHEMAS_TABLE_PATH = Path("./tables/schemas.csv")
TRACKER_TABLE_PATH = Path("tracker.json")
DATA_DIR_PATH = Path("./plan_finder_data")


@contextmanager
def create_data_dir(*, delete_on_finish: bool = True) -> bool:
    """Temporarily create a data directory and delete it when done."""
    # Create data dir for zip files
    DATA_DIR_PATH.mkdir(exist_ok=True)
    logger.info(f"Created temporary dir {DATA_DIR_PATH}")
    try:
        yield
    finally:
        # Clean up data dir on completion, even if there was an error
        if delete_on_finish is True:
            shutil.rmtree(DATA_DIR_PATH)
            logger.info(f"Deleted temporary dir {DATA_DIR_PATH}")


class Loader:
    """A class that loads Socrata datasets for a particular env."""

    env: Literal["test", "impl", "prod"]
    files_table: Dict[str, Tuple[str, str]]
    schemas_table: Dict[str, List[str]]
    tracker_table: Dict[str, str]
    access_token: str = None
    access_token_expires: DateTime = None
    client: Socrata

    def __init__(self, env: Literal["test", "impl", "prod"]) -> None:
        """Initialize this class instance with default values."""
        logger.info(f"Initializing Loader for env {env}")
        # Set env
        self.env = env

        # Load lookup tables
        self.load_files_table()
        self.load_schemas_table()
        self.load_tracker_table()

        # Initialize Socrata client
        auth: Tuple[str] = Authorization(SOCRATA_DOMAIN, *SOCRATA_CREDENTIALS)
        self.client = Socrata(auth)

    @staticmethod
    def _request_details(r) -> None:
        """given a response, return a string showing the status code, headers, and body"""
        request_headers = "\n" + "  \n".join(f"{k}: {v}"
                                             for k, v in r.request.headers.items())
        response_headers = "\n" + "  \n".join(f"{k}: {v}"
                                              for k, v in r.headers.items())
        return f"""requested {r.request.url}
request headers:{request_headers}
request cookies: {r.request._cookies}
request body: {r.request.body}

response code: {r.status_code}
response headers:{response_headers}
body: {r.text}
"""

    def load_files_table(self) -> None:
        """Load the files table as a dict on this instance."""
        table: Dict[str, Tuple[str, str]] = {}
        with open(FILES_TABLE_PATH) as files_table_file:
            reader: Iterable[List[str]] = csv.reader(files_table_file)
            next(reader)
            for file_name, file_encoding in reader:
                table[file_name] = file_encoding
        self.files_table = table
        logger.info(f"Loaded files table: {FILES_TABLE_PATH}")

    def load_schemas_table(self) -> None:
        """Load the schemas table as a dict on this instance."""
        table_unsorted: Dict[str, Tuple[int, str]] = {}
        with open(SCHEMAS_TABLE_PATH) as schemas_table_file:
            reader: Iterable[List[str]] = csv.reader(schemas_table_file)
            next(reader)
            for file_name, column_index, column_name in reader:
                table_unsorted.setdefault(file_name, [])
                table_unsorted[file_name].append((int(column_index), column_name))

        # Sort columns by column index
        table: Dict[str, List[str]] = {}
        for file_name, columns in table_unsorted.items():
            columns_sorted: Iterable[Tuple[int, str]] = sorted(columns, key=itemgetter(0))
            column_names: List[str] = [column_name for column_index, column_name in columns_sorted]
            table[file_name] = column_names
        self.schemas_table = table
        logger.info(f"Loaded schemas table: {SCHEMAS_TABLE_PATH}")

    def load_tracker_table(self) -> None:
        """Load the tracker table as a dict on this instance."""
        with open(TRACKER_TABLE_PATH) as tracker_table_file:
            table: Dict[str, str] = json.load(tracker_table_file)
        self.tracker_table = table
        logger.info(f"Loaded tracker table: {TRACKER_TABLE_PATH}")

    def update_tracker_table(self) -> None:
        """Update the tracker table and write it back to disk."""
        logger.info(f"Updating tracker table: {TRACKER_TABLE_PATH}")
        with open(TRACKER_TABLE_PATH, "w") as tracker_table_file:
            json.dump(self.tracker_table, tracker_table_file, indent=2, sort_keys=True)
        logger.info(f"Updated tracker table: {TRACKER_TABLE_PATH}")

    def fetch_access_token(self) -> None:
        """Fetch an access token to obtain Plan Finder data."""
        # Construct request
        url: str = TOKEN_URLS[self.env]
        username: str = USERNAMES[self.env]
        key_id, key_secret = API_KEYS[self.env]
        body = {
            "userName": username,
            "scopes": "mpfpe_pde_full",
            "keyId": key_id,
            "keySecret": key_secret,
        }
        params = {}
        if self.env in ACS_PARAMS:
            params["ACS"] = ACS_PARAMS[self.env]

        # Submit HTTP POST request to obtain token
        logger.info(f"Fetching {self.env} access token")
        response: Response = requests.post(url, json=body, params=params)
        if response.status_code != 200:
            logger.error(Loader._request_details(response))
            raise RuntimeError(f"Failed to fetch token: HTTP status {response.status_code}")

        # Extract token from response
        response_json: dict = response.json()
        access_token: str = response_json["accessToken"]
        expires: int = response_json["expires"]
        self.access_token = access_token
        self.access_token_expires = DateTime.now() + Duration(seconds=expires)
        logger.info(f"Fetched {self.env} access token; expires {self.access_token_expires}")

    def fetch_zip_file(self, plan_year: str, date: Date = Date.today()) -> Path:
        """Download a Plan Finder zip file for a given date."""
        # If we don't have a current access token, fetch one
        no_access_token = self.access_token is None
        if no_access_token or DateTime.now() > (self.access_token_expires - Duration(minutes=5)):
            self.fetch_access_token()

        # Construct request
        url = DATA_URL
        headers = {
            "X-API-CONSUMER-ID": API_KEYS[self.env][0],
            "Authorization": f"Bearer {self.access_token}",
        }
        params = {"fileName": f"{plan_year}_{date.to_date_string()}"}

        # Submit GET request to download file
        logger.info(f"Fetching {self.env} zip file for plan year {plan_year} and date {date}")
        response = requests.get(url, headers=headers, params=params)
        if not response.status_code == 200:
            raise RuntimeError(
                "Failed to fetch zip file (this may be expected for dates with no data): HTTP "
                f"status {response.status_code}"
            )

        # Save zip file to disk and return its path
        zip_bytes: bytes = response.content
        zip_file_path = DATA_DIR_PATH / f"{self.env}_{date}.zip"
        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(zip_bytes)
        logger.info(f"Fetched {self.env} zip file: {zip_file_path}")
        return zip_file_path

    def unzip_zip_file(self, zip_file_path: Path) -> Path:
        """Unzip a zip file on disk.

        Because Plan Finder zip files throw errors when trying to unzip
        with a newer zip utility such as Python's built-in zipfile
        module, we need to use subprocess.run to call the unzip
        executable as a workaround.
        """
        extract_dir_path: Path = zip_file_path.parent / zip_file_path.stem
        logger.info(f"Unzipping zip file: {zip_file_path}")
        result: subprocess.CompletedProcess = subprocess.run(
            ["unzip", str(zip_file_path), "-d", str(extract_dir_path)]
        )
        if result.returncode == 0:
            logger.info(f"Unzipped zip file {zip_file_path}: {extract_dir_path}")
            return extract_dir_path
        else:
            raise RuntimeError(f"Failed to unzip {zip_file_path}: exit code {result.returncode}")

    def load_dataframe(self, data_file_path: Path) -> pd.DataFrame:
        """Load a pandas DataFrame object for a Plan Finder data file.

        This is the most robust/simple way of handling the various
        encoding and schema issues associated with Plan Finder data.
        """
        # Look up the schema (column indices and names) matching this file
        try:
            file_encoding = self.files_table[data_file_path.name]
        except KeyError:
            raise KeyError(
                f"Failed to find encoding for {data_file_path.name} in {FILES_TABLE_PATH}"
            )

        # Look up column names from schemas table
        try:
            column_names: List[str] = self.schemas_table[data_file_path.name]
        except KeyError:
            raise KeyError(
                f"Failed to find schema for {data_file_path.name} in {SCHEMAS_TABLE_PATH}"
            )

        # Load file as a dataframe using the column names and encoding we identified
        dataframe: pd.DataFrame = pd.read_csv(
            data_file_path, names=column_names, encoding=file_encoding, delimiter="\t", dtype=str
        )
        return dataframe

    def prepare_output_schema(self, output_schema: OutputSchema) -> OutputSchema:
        """Set all columns to text within a Socrata-py OutputSchema."""
        columns: List[dict] = output_schema.attributes["output_columns"]
        for column in columns:
            # Extract both field name and transform expression version of field name
            column_name: str = column["field_name"]
            transform_expr: str = column["transform"]["transform_expr"]
            column_name_match: re.Match = re.search(r"`[^`]+`", transform_expr)
            transform_column_name: str = column_name_match.group(0)
            transform = f"to_text({transform_column_name})"
            output_schema = output_schema.change_column_transform(column_name).to(transform)
        changed_output_schema: OutputSchema = output_schema.run()
        return changed_output_schema

    def is_old_hanging_draft(self, revision: dict) -> bool:
        """Determine whether a draft is at least a day old and unclosed."""
        draft_created_at: DateTime = pendulum.parse(revision["resource"]["created_at"])
        is_old_draft: bool = DateTime.now() - draft_created_at > Duration(days=1)
        is_hanging_draft: bool = revision["resource"]["closed_at"] is None
        return is_old_draft and is_hanging_draft

    def delete_old_hanging_drafts(self, dataset_id: str) -> None:
        """Delete any old hanging drafts for a given dataset.

        Checks the dataset's revision history for unclosed drafts that
        are at least a day old, and deletes any that are found.
        """
        # Check for hanging drafts
        auth: Tuple[str, str] = (self.client.auth.username, self.client.auth.password)
        domain_url = f"https://{self.client.auth.domain}".rstrip("/")
        revisions_url = f"{domain_url}/api/publishing/v1/revision/{dataset_id}"
        logger.info(f"Checking for old hanging drafts for dataset {dataset_id}…")
        response: Response = requests.get(revisions_url, auth=auth)
        try:
            response_json: List[dict] = response.json()
            old_hanging_drafts: Iterable[dict] = filter(self.is_old_hanging_draft, response_json)
            discard_urls: Iterable[str] = map(
                lambda revision: revision["links"]["discard"], old_hanging_drafts
            )
        except Exception:
            logger.exception(
                f"Failed when trying to delete old hanging drafts for dataset {dataset_id}"
            )
            return

        # Delete hanging drafts in turn
        for discard_url in discard_urls:
            delete_url = f"{domain_url}{discard_url}"
            logger.info(f"Deleting old hanging draft: {delete_url}")
            delete_response: Response = requests.delete(delete_url, auth=auth)
            delete_status: int = delete_response.status_code
            logger.info(f"Obtained HTTP {delete_status} response: {delete_url}")

    def create_dataset(self, data_file_path: Path, date: Date) -> str:
        """Create a new dataset on Socrata from a data file."""
        # Load dataframe
        dataframe: pd.DataFrame = self.load_dataframe(data_file_path)
        dataset_name = f"{data_file_path.stem} [{self.env}]"
        description = f"Plan Finder dataset {dataset_name}, released on {date}."

        # Create new dataset on Socrata, set all columns as text, and publish
        logger.info(f"Creating new dataset on Socrata: {data_file_path}")
        revision: Revision
        output_schema: OutputSchema
        revision, output_schema = self.client.create(
            name=dataset_name, description=description, category="Plan Finder", tags=[self.env]
        ).df(dataframe)
        revision = revision.update({"action": {"permission": SOCRATA_DATASET_PERMISSION}})
        dataset_id: str = revision.attributes["fourfour"]
        output_schema.wait_for_finish()
        output_schema = self.prepare_output_schema(output_schema)
        output_schema.wait_for_finish()
        job: Job = revision.apply()
        job.wait_for_finish()
        logger.info(f"Created dataset: {dataset_id}")
        return dataset_id

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def update_dataset(self, dataset_id: str, data_file_path: Path, date: Date) -> str:
        """Update an existing dataset on Socrata from a data file.

        If a failure occurs when updating, this function will attempt a single
        retry; if the failure persists on retry, the exception will be caught
        and logged in Loader.update_all_datasets.
        """
        # Delete old hanging drafts for this dataset (e.g. from previous failures)
        self.delete_old_hanging_drafts(dataset_id)

        # Load dataframe
        dataframe: pd.DataFrame = self.load_dataframe(data_file_path)
        dataset_name = f"{data_file_path.stem} [{self.env}]"
        description = f"Plan Finder dataset {dataset_name}, released on {date}."

        # Create replace revision on Socrata and publish
        logger.info(f"Updating dataset {dataset_id} on Socrata: {data_file_path}")
        view: View = self.client.views.lookup(dataset_id)
        revision: Revision = view.revisions.create_replace_revision(
            metadata={"description": description}, permission=SOCRATA_DATASET_PERMISSION
        )
        upload: Source = revision.create_upload(data_file_path.name)
        source: Source = upload.df(dataframe)
        source.wait_for_finish()
        output_schema: OutputSchema = source.get_latest_input_schema().get_latest_output_schema()
        output_schema.wait_for_finish()
        revision.apply(output_schema=output_schema)
        logger.info(f"Updated dataset: {dataset_id}")
        return dataset_id

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete an existing dataset on Socrata."""
        logger.info(f"Deleting dataset {dataset_id} on Socrata")
        auth: Tuple[str, str] = (self.client.auth.username, self.client.auth.password)
        domain_url = f"https://{self.client.auth.domain}".rstrip("/")
        dataset_url = f"{domain_url}/api/views/{dataset_id}"
        response: Response = requests.delete(dataset_url, auth=auth)
        response.raise_for_status()
        logger.info(f"Deleted dataset: {dataset_id}")
        return True

    def create_all_datasets(self, plan_year: str, date: Date = Date.today(), only_untracked: bool = False) -> None:
        """Create Socrata datasets for Plan Finder data for a given date."""
        logger.info(f"Creating all {self.env} datasets on Socrata for {date}")
        self.load_tracker_table()
        if only_untracked is not True and any(self.tracker_table[self.env].values()):
            raise RuntimeError("Please delete all datasets in tracker before creating new ones")

        with create_data_dir():
            # Fetch and unzip zip file for this env and date
            zip_file_path: Path = self.fetch_zip_file(plan_year, date)
            extract_dir_path: Path = self.unzip_zip_file(zip_file_path)

            # Iterate over all files in newly unzipped directory
            data_file_paths: Iterable[Path] = extract_dir_path.glob("*")
            for data_file_path in data_file_paths:
                # Skip files that are not in the files table
                if data_file_path.name not in self.files_table:
                    continue

                # When only_untracked is True, skip files for which a dataset is already tracked
                dataset_is_tracked = (
                    self.tracker_table[self.env].get(data_file_path.name) is not None
                )
                if only_untracked is True and dataset_is_tracked:
                    continue

                # Create new dataset, skipping this one if we get an error
                try:
                    dataset_id: str = self.create_dataset(data_file_path, date)
                except Exception:
                    logger.exception(f"Failed to create dataset: {data_file_path}")
                    continue
                else:
                    # Add new dataset identifier (4x4) to tracker
                    self.tracker_table[self.env][data_file_path.name] = dataset_id
            logger.info(f"Finished creating {self.env} datasets for {date}")
            self.update_tracker_table()

    def update_all_datasets(
            self, plan_year: str, date: Date = Date.today(), only_file: Optional[str] = None
    ) -> None:
        """Update existing Socrata Plan Finder datasets for a given date."""
        logger.info(f"Updating {self.env} datasets on Socrata for {date}")
        self.load_tracker_table()
        with create_data_dir():
            # Fetch and unzip zip file for this env and date
            zip_file_path: Path = self.fetch_zip_file(plan_year, date)
            extract_dir_path: Path = self.unzip_zip_file(zip_file_path)

            # Iterate over all existing datasets in tracker
            failed_updates: List[Tuple[str, str]] = []
            for file_name, dataset_id in self.tracker_table[self.env].items():
                data_file_path = extract_dir_path / file_name

                # If only_file is specified, skip files not matching the supplied filename
                if only_file is not None and data_file_path.name != only_file:
                    continue

                # Skip files not included in this date's release or not in files table
                if not data_file_path.exists() or data_file_path.name not in self.files_table:
                    continue

                # Update dataset, skipping this one if we get an error
                try:
                    dataset_id: str = self.update_dataset(dataset_id, data_file_path, date)
                except Exception:
                    logger.exception(f"Failed to update dataset {dataset_id}: {data_file_path}")
                    failed_updates.append((file_name, dataset_id))
                    continue
            logger.info(
                f"Finished updating {self.env} datasets for {date}; {len(failed_updates)} failures"
            )
            for file_name, dataset_id in failed_updates:
                logger.info(f"Failed to update {file_name} [{self.env}] ({dataset_id})")

    def delete_all_datasets(self) -> None:
        """Delete all existing Plan Finder datasets on Socrata."""
        logger.info(f"Deleting all {self.env} datasets on Socrata")
        self.load_tracker_table()

        # Iterate over all existing datasets in tracker
        for file_name, dataset_id in self.tracker_table[self.env].items():
            # Skip blank dataset IDs
            if not dataset_id:
                continue

            # Update dataset, skipping this one if we get an error
            try:
                self.delete_dataset(dataset_id)
            except Exception:
                logger.exception(f"Failed to delete dataset {dataset_id}")
                continue
            else:
                self.tracker_table[self.env][file_name] = ""
        logger.info(f"Finished deleting {self.env} datasets")
        self.update_tracker_table()


def parse_command_line_args() -> Namespace:
    """Initialize a parser and parse any command line arguments."""
    # Create basic command line interface
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        type=lambda date: pendulum.parse(date, exact=True),
        default=Date.today().to_date_string(),
        help=f"date to retrieve, in YYYY-MM-DD format; defaults to today: {Date.today()}",
    )
    parser.add_argument(
        "--env",
        dest="envs",
        action="append",
        choices=ENVS,
        default=[],
        help="HPMS environment to use (test, impl, or prod); defaults to all three",
    )
    parser.add_argument(
        "--only-file",
        metavar="FILENAME",
        help="filename to update; if specified, only that file (and no others) will be updated",
    )
    parser.add_argument(
        "--create-untracked-datasets",
        action="store_true",
        help=(
            "create any datasets listed in CSV tables but not currently tracked in tracker.json; "
            "if this flag is set, the script will only create the untracked datasets, but not "
            "perform any other updates"
        ),
    )
    parser.add_argument(
        "--plan-year",
        nargs=1,
        dest="plan_year",
        help="Plan year to fetch. Required",
    )
    return parser.parse_args()


def main() -> None:
    """Execute this script to update or create Socrata datasets."""
    command_line_args: Namespace = parse_command_line_args()
    start_time: DateTime = DateTime.now()
    if not command_line_args.envs:
        command_line_args.envs = ENVS
    envs_summary = ", ".join(command_line_args.envs)
    date: Date = command_line_args.date
    plan_year: str = command_line_args.plan_year

    if not plan_year:
        raise Exception("Missing required argument --plan-year")

    # Create untracked datasets if --create-untracked-datasets is specified
    if command_line_args.create_untracked_datasets is True:
        logger.info("Not updating datasets because --create-untracked-datasets was specified")
        if command_line_args.only_file:
            logger.info("Ignoring --only-file because --create-untracked-datasets was specified")
        logger.info(f"Creating untracked datasets using data from {date}, envs: {envs_summary}")
        for env in command_line_args.envs:
            logger.info(f"Loading env: {env}")
            loader = Loader(env)
            loader.create_all_datasets(plan_year, only_untracked=True)
    # Otherwise, just update all datasets (default behavior)
    else:
        logger.info(f"Updating datasets for {date}, envs: {envs_summary}")
        for env in command_line_args.envs:
            logger.info(f"Loading env: {env}")
            loader = Loader(env)
            loader.update_all_datasets(plan_year, date, only_file=command_line_args.only_file)

    time_elapsed: Duration = DateTime.now() - start_time
    logger.info(f"Finished! Time elapsed: {time_elapsed.in_words()}")


# Run main() if we're executing this script on the command line; if we are importing a function or
# class from another Python process, this prevents inadvertently running the entire script
if __name__ == "__main__":
    main()
