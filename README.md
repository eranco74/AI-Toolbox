# AI-Toolbox
A set of AI-driven tools designed to assist in creating, training, and deploying AI agents across different workflows.


Getting started:

```shell
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

Define a few environment variables in the envs file:
``` shell
LLM_ENDPOINT=
GITHUB_TOKEN=
REPO_NAME=
PR_NUM=
# Optional
MODEL=
API_KEY=
```

 Run the github-pr-review-agent
```shell
source ./envs
./github-pr-review-agent.p
```
