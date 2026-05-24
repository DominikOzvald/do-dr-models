import argparse
import os
import requests


def fetch_log(token: str, job_name: str, repo: str, run_id: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    print(f"Looking for job {job_name}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        jobs = response.json()["jobs"]
        job_id = None
        for job in jobs:
            if job["name"] == job_name:
                job_id = job["id"]
                print(f"Found job {job_name}")
        if job_id:
            url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(f"{job_name}.txt", "w") as f:
                    f.write(response.text)
                print(f"Successfully loaded logs to file  {job_name}.txt")
            else:
                print(
                    f"Failed to load logs. Response status code {response.status_code}"
                )
                exit(-1)
        else:
            print(f"Could not find job {job_name}")
            exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--token", required=True, type=str)
    parser.add_argument("-j", "--job", required=True, type=str)

    args = parser.parse_args()
    token = args.token
    job = args.job
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    fetch_log(token, job, repo, run_id)
