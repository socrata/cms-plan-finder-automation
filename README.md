# upload_plan_finder_data.py

The Python script `upload_plan_finder_data.py` can be used to perform automated daily updates to CMS Plan Finder data on Socrata.

The CMS Plan Finder data can be found on this Socrata domain:

    https://plan-finder.data.medicare.gov

This file contains documentation on technical requirements for running the script as well as a quick guide to scheduling the script for data automation.

## Requirements

This script requires a recent version of Python (3.6+), as well as some external dependencies which can be installed using the requirements.txt file like so:

    pip3 install -r requirements.txt

### Environment variables

Additionally, the script uses a number of credentials which must be securely stored using environment variables. (It's best to avoid inserting private credentials in the script itself.) Here are the required environment variables:

    # Required to obtain access token from the HPMS token services
    SOCRATA_PLAN_FINDER_TEST_USERNAME
    SOCRATA_PLAN_FINDER_IMPL_USERNAME
    SOCRATA_PLAN_FINDER_PROD_USERNAME
    SOCRATA_PLAN_FINDER_TEST_KEY_ID
    SOCRATA_PLAN_FINDER_TEST_KEY_SECRET
    SOCRATA_PLAN_FINDER_IMPL_KEY_ID
    SOCRATA_PLAN_FINDER_IMPL_KEY_SECRET
    SOCRATA_PLAN_FINDER_PROD_KEY_ID
    SOCRATA_PLAN_FINDER_PROD_KEY_SECRET
    SOCRATA_PLAN_FINDER_TEST_ACS
    SOCRATA_PLAN_FINDER_IMPL_ACS

    # Required to log into the Socrata domain and update datasets
    SOCRATA_KEY_ID
    SOCRATA_KEY_SECRET

### Local files

The script also depends on a few local files to understand which encodings, schemas (column names), and Socrata dataset IDs to use for which Plan Finder data files. Do not modify or delete these unless you know what you're doing:

    tables/files.csv
    tables/schemas.csv
    tracker.json

## Getting Started

To automate Plan Finder data using this script, use a scheduling service such as Windows Task Scheduler (on Windows) or cron (on Linux). Here's the command to automate:

    python3 upload_plan_finder_data.py

Since the Plan Finder data is released daily, schedule the script to run no more frequently than once a day. The script will attempt to download a file using the current day's date (local time), so be careful about scheduling it to run in the early morning, in case the new data hasn't yet been released.

## How it works

Each time it's run, the script executes the following steps:

1. Load the local tables `tables/files.csv`, `tables/schemas.csv`, and `tracker.json`. These files contain the file encodings, schemas, and Socrata dataset IDs required to automate the Plan Finder data.
2. Fetch an HPMS access token for the `test` environment.
3. Create a temporary directory, `plan_finder_data`, for storing today's data files.
4. Fetch the Plan Finder zip file for the current date in the HPMS `test` environment.
5. Extract the contents of the zip file in the `plan_finder_data` directory.
6. For each extracted data file…
    * Identify an encoding, schema, and Socrata dataset ID from the local tables loaded in step 1. (If these are not found, skip the remaining steps.)
    * Cleanup: Check for old draft updates to this dataset on Socrata that were created more than 1 day ago but never successfully finished publishing. Discard these hanging drafts.
    * Load the data file using the specified encoding and schema (column names).
    * Using the specified Socrata dataset ID, upload the new data file, fully replacing the rows in the existing dataset.
7. If any updates from step 6 failed, log a summary of the failures.
8. Delete the temporary directory `plan_finder_data` that was created in step 3.
9. Repeat steps 1-8 for the `impl` and `prod` environments.

## Options

The script supports a few command line options. Here's the full command line interface (you can also get this by executing `python3 upload_plan_finder_data.py --help`):

    python3 upload_plan_finder_data.py [-h] [--date YYYY-MM-DD]
                                    [--env {test,impl,prod}]
                                    [--only-file FILENAME]
                                    [--create-untracked-datasets]

    A script to automate updates to CMS Plan Finder data on Socrata.

    Requires a number of environment variables to be set prior to script
    execution. See the README.md file for more information.

    optional arguments:
    -h, --help            show this help message and exit
    --date YYYY-MM-DD     date to retrieve, in YYYY-MM-DD format; defaults to today: 2020-07-30
    --env {test,impl,prod}
                            HPMS environment to use (test, impl, or prod); defaults to all three
    --only-file FILENAME  filename to update; if specified, only that file (and no others) will be updated
    --create-untracked-datasets
                            create any datasets listed in CSV tables but not currently tracked in tracker.json; if this flag is set, the script will only create the untracked datasets, but not perform any other updates

## Logging

The script will log its output to the console, as well as to a local file, `socrata.log`. These logs are the best place to start if you're troubleshooting an error.

## How-tos

### Updating datasets based on a particular date

Let's say we want to update all datasets with the data released on a particular date, such as April 24, 2020. We can do so using the `--date` option:

    python3 upload_plan_finder_data.py --date 2020-04-24

### Updating datasets within a particular environment

What if we only want to update datasets for the `test` environment, but not `impl` or `prod`? We can use the `--env` option:

    python3 upload_plan_finder_data.py --env test

### Updating only datasets associated with a specific data file

In this example, we want to update only the datasets associated with the Plan Finder data file `PLN_BNFT_PKG_CST.TXT`, but not any others. We can do this using the `--only-file` option:

    python3 upload_plan_finder_data.py --only-file PLN_BNFT_PKG_CST.TXT

### Combining options

We can also combine options to update only the dataset for a specific file, date, and environment:

    python3 upload_plan_finder_data.py --only-file PLN_BNFT_PKG_CST.TXT --date 2020-04-24 --env test

### Adding new data files

What if a new data file becomes available in the HPMS release package? How do we add it to our automation?

Because Plan Finder data files do not include column names (schemas), some manual configuration is required. Follow these steps:

1. **Add row to files table:** Add a comma-delimited row to `tables/files.csv` containing the new file's **name** and **encoding**. (The encoding must be a valid Python-readable encoding string such as `utf-16-le` or `ascii`.) This ensures that the script will find your file and open it using the correct encoding.

2. **Add row to schemas table:** Add comma-delimited rows to `tables/schemas.csv` containing the file's **name** and **column indices and names**. The number and order of columns in the schemas table must match the data file exactly.

3. **Create untracked datasets on Socrata:** Open a terminal and execute the script with the `--create-untracked-datasets` option: `python3 upload_plan_finder_data.py --create-untracked-datasets`.

4. **Check the Plan Finder domain:** If the script executes successfully, your new dataset(s) should be created on the Plan Finder domain. You can find their unique Socrata identifiers by inspecting the `tracker.json` file. (As mentioned above, `tracker.json` is the file that links Plan Finder data files to their respective Socrata datasets.) Congratulations! Your new data file(s) are now tracked by the script and will be automatically updated when the script runs each day.
