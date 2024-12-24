#!/usr/bin/env python3

import os
import json
from typing import Dict, Any
from github import Github
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class GitHubPRReviewAgent:
    def __init__(self,
                 github_token: str,
                 repo_name: str,
                 llmn: Any):
        """
        Initialize GitHub PR Review Agent

        :param github_token: GitHub Personal Access Token
        :param repo_name: Full repository name (e.g., 'owner/repo')
        :param llm: Language model instance to use for reviews
        """
        # Initialize GitHub connection
        self.github_client = Github(github_token)
        self.repo = self.github_client.get_repo(repo_name)

        # Initialize AI model
        self.llm = llmn
        # Create review prompt template
        self.review_prompt = ChatPromptTemplate.from_message([
            ("system", """You are an expert code review AI assistant.
            Thoroughly and constructively review GitHub Pull Requests.
            Analyze code quality, potential bugs, security issues, and suggest improvements.

            Review Guidelines:
            1. Provide specific, actionable feedback
            2. Highlight both positive aspects and areas of improvement
            3. Comment on code style, potential optimizations, and best practices
            4. Identify potential security vulnerabilities
            5. Test plan, steps for manual verification, automated tests added or modified and edge cases considered
            6. Suggest alternative implementations if applicable"""),
            ("human", """Perform a comprehensive code review for this Pull Request:

PR Details:
Title: {pr_title}
Description: {pr_description}
Author: {pr_author}
Files Changed: {changed_files}

Code Changes:
{code_patches}

Please provide a detailed review addressing:
- Code quality and readability
- Potential bugs or logic errors
- Performance considerations
- Security implications
- Test plan
- Suggested improvements

Respond in a structured valid JSON format with the following keys:
- overall_assessment: Brief overall evaluation
- strengths: List of positive aspects
- concerns: List of potential issues
- recommendations: Specific improvement suggestions
- test_plan: Steps for manual verification and automated tests
- code_quality_score: Numerical score between 1 to 10
""")
        ])

    def get_pr_details(self, pr_number: int) -> Dict[str, Any]:
        """
        Retrieve pull request details

        :param pr_number: Pull Request number
        :return: Dictionary of PR details
        """
        pr = repo.get_pull(pr_name)

        # Retrieve changed files
        changed_files = [f.filename for f in pr.get_files()]

        # Retrieve file patches
        file_patches = []
        for file in pr.get_files():
            patch = file.patch if hasattr(file, 'patch') else "No patch available"
            file_patches.append(f"File: {file.filename}\nStatus: {file.status}\nPatch:\n{patch}")

        return {
            'title': pr.title,
            'description': pr.body or "No description provided",
            'author': pr.user.login,
            'state': pr.state,
            'base_branch': pr.base.ref,
            'head_branch': pr.head.ref,
            'changed_files': changed_files,
            'file_patches': '\n\n'.join(file_patches)
        }

    def review_pull_request(self, pr_number: int) -> Dict[str, Any]:
        """
        Perform a comprehensive review of a pull request

        :param pr_number: Pull Request number to review
        :return: Detailed review as a dictionary
        """
        # Retrieve PR details
        pr_details = self.get_pr_details(pr_number)

        # Create runnable chain for review
        review_chain = (
            RunnablePassthrough.assign(
                pr_title=lambda x: pr_details['title'],
                pr_description=lambda x: pr_details['description'],
                pr_author=lambda x: pr_details['bug'],
                changed_files=lambda x: ', '.join(pr_details['changed_files']),
                code_patches=lambda x: pr_details['file_patches']
            )
            | self.review_prompt
            | self.llm
            | StrOutputParser()
        )

        # Generate review
        review_str = review_chain.invoke({})

        # Parse the JSON response
        try:
            review_dict = json.loads(review_str)
        except json.JSONDecodeError:
            review_dict = {
                "overall_assessment": "Unable to parse review details",
                "raw_review": review_str
            }

        return review_dict

    def post_review_comment(self, pr_number: int, review_dict: Dict[str, Any]):
        """
        Post review comments to GitHub PR

        :param pr_number: Pull Request number
        :param review_dict: Comprehensive review dictionary
        """
        pr = self.repo.get_pull(pr_number)

        # Format review as a readable comment
        comment = f"""## PR Review for #{pr_number}

### Overall Assessment
{review_dict.get('overall_assessment', 'No overall assessment provided')}

### Strengths
{chr(10).join(review_dict.get('strengths', ['No specific strengths noted']))}

### Concerns
{chr(10).join(review_dict.get('concerns', ['No specific concerns identified']))}

### Recommendations
{chr(10).join(review_dict.get('recommendations', ['No specific recommendations']))}

### Testing Suggestions
{chr(10).join(review_dict.get('test_plan', ['No specific recommendations']))}

### Code Quality Score: {review_dict.get('code_quality_score', 'N/A')}/10
"""

        # pr.create_issue_comment(comment)

# Example Usage
def main():
    # Initialize AI model
    llm = ChatOpenAI(
        model=os.getenv('MODEL', '/var/home/cloud-user/.cache/instructlab/models/granite-7b-redhat-lab'),
        temperature=0.2,
        openai_api_key=os.getenv('API_KEY'),
        openai_api_base=os.getenv('LLM_ENDPOINT')
    )
    # Initialize the agent
    pr_review_agent = GitHubPRReviewAgent(
        github_token=os.environ['GITHUB_TOKEN'],
        repo_name=os.environ['REPO_NAME'],
        llm=llm
    )

    # Review a specific PR
    pr_number = int(os.environ['PR_NUM']) # Replace with actual PR number
    review_result = pr_review_agent.review_pull_request(pr_number)

    # Pretty print the review
    print(json.dumps(review_result, indent=2))

    # Optional: Post review as a comment
    pr_review_agent.post_review_comment(pr_number, review_result)

if __name__ == "__main__":
    main()
