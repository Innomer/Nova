from github import Github
import requests

class GitHubClient:
    def __init__(self, username, password):
        self.github = Github(username, password)

    def get_repository(self, repository_name):
        return self.github.get_repo(repository_name)

    def push_changes(self, repository, branch_name):
        branch = repository.get_branch(branch_name)
        repository.git.push("origin", branch.name)

    def get_commits(self, repository, branch_name):
        branch = repository.get_branch(branch_name)
        commits = branch.commit
        return commits

    def commit_changes(self, repository, branch_name, file_paths, commit_message):
        branch = repository.get_branch(branch_name)
        parent_commit = repository.get_commit(branch.commit.sha)
        tree = repository.get_git_tree(parent_commit.sha)
        new_tree = []
        
        for entry in tree.tree:
            if entry.path not in file_paths:
                new_tree.append(entry)
        
        for path in file_paths:
            with open(path, 'r') as file:
                data = file.read()
            new_file = repository.create_file(path, commit_message, data, branch=branch_name)
            new_tree.append(new_file)

    def reset_branch(self, repository, branch_name):
        branch = repository.get_branch(branch_name)
        repository.git.reset("--hard", f"origin/{branch_name}")

    def pull_changes(self, repository, branch_name):
        self.download_files(repository)

    def download_files(self, repository, path='/'):
        contents = repository.get_contents(path)
        for content in contents:
            if content.type == "file":
                file = requests.get(content.download_url)
                with open(repository.name + "/" + content.name, "wb") as f:
                    f.write(file.content)
            elif content.type == "dir":
                self.download_files(repository, content.path)

    def get_repositories(self):
        return self.github.get_user().get_repos(visibility="all")

# Example usage
if __name__ == "__main__":
    github_username = ""
    github_password = ""
    repository_name = input("Enter the name of the repository: ")
    repository_name = "Innomer/" + repository_name
    branch_name = input("Enter the name of the branch: ")

    github_client = GitHubClient(github_username, github_password)
    repositories = github_client.get_repositories()
    for repo in repositories:
        print(repo.name)

    repository = github_client.get_repository(repository_name)
    print(repository)

    # Example usage of functions
    # github_client.push_changes(repository, branch_name)
    commits = github_client.get_commits(repository, branch_name)
    print(commits)
    # github_client.commit_changes(repository, branch_name, file_paths=[r"C:\MY FILES\CSX\TechTitans_CSX\desktop_app\file.txt"], commit_message="Commit message")
    # github_client.reset_branch(repository, branch_name)
    # github_client.pull_changes(repository, branch_name)
