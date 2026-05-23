import os

import requests
import zipfile
import io
PAT_TOKEN = ""


def get_n_logs(token, repo, owner, n,out_dir):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    runs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    params = {"per_page": n}

    response = requests.get(runs_url, params=params, headers=headers)
    if response.status_code ==200:
        runs = response.json()["workflow_runs"]

        for run in runs:
            run_id = run["id"]
            run_number = run["run_number"]
            workflow_name = run["name"]
            run_dir = os.path.join(out_dir,f"{workflow_name}-{run_number}")
            os.makedirs(run_dir,exist_ok=True)
            print(f"Saving run {workflow_name}-{run_number} to directory {run_dir}")

            run_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
            response = requests.get(run_url,headers=headers)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as file:
                    file.extractall(run_dir)
                print(f"Successfully saved run {workflow_name}-{run_number}")

    else:
        print(f"Run status code: {response.status_code}")


if __name__ == "__main__":
    repo = "dummy"
    owner = "DominikOzvald"
    n = 22
    get_n_logs(PAT_TOKEN,repo,owner,n,out_dir="C:/Faks/Diplomski rad/data/dummy")

