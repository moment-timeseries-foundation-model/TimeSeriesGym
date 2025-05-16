import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import requests
from get_discussion_posts import extract_text_from_div
from git import Repo
from newspaper import Article
from playwright.sync_api import sync_playwright
from readability import Document


def detect_type(resource: str) -> str:
    """
    Detect resource type by URL or filesystem path.
    Returns: 'github', 'arxiv', or 'webpage'.
    """
    # GitHub URL
    if resource.startswith("git@github.com") or resource.startswith("https://github.com"):
        return "github"
    # arXiv PDF or abs link
    if re.search(r"arxiv\.org/(abs|pdf)", resource):
        return "arxiv"
    # File path to local PDF or LaTeX
    if os.path.exists(resource) and resource.lower().endswith(".pdf"):
        return "arxiv"
    if "kaggle" in resource:
        return "kaggle"
    # Default to webpage
    return "webpage"


class RepoHandler:
    def __init__(self, url, clone_dir=None, readme_only=False):
        self.url = url
        self.clone_dir = clone_dir or tempfile.mkdtemp()
        self.readme_only = readme_only

    def fetch(self):
        Repo.clone_from(self.url, self.clone_dir, depth=1)
        return self.clone_dir

    def extract_text(self):
        texts = []
        # README and docs
        for root, _, files in os.walk(self.clone_dir):
            for fn in files:
                if fn.lower().startswith("readme") or fn.endswith((".md", ".rst", ".txt")):
                    path = os.path.join(root, fn)
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())

        if not self.readme_only:
            # Comments from code
            for root, _, files in os.walk(self.clone_dir):
                for fn in files:
                    if fn.endswith((".py", ".java", ".js", ".cpp", ".c", ".go")):
                        path = os.path.join(root, fn)
                        with open(path, encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if (
                                    line.startswith("#")
                                    or line.startswith("//")
                                    or line.startswith("/*")
                                ):
                                    texts.append(line)
        return "\n".join(texts)

    def clean_up(self):
        shutil.rmtree(self.clone_dir, ignore_errors=True)


class WebpageHandler:
    def __init__(self, url):
        self.url = url

    def fetch(self):
        resp = requests.get(self.url, timeout=30)
        resp.raise_for_status()
        return resp.text

    def extract_text(self, html=None):
        html = html or self.fetch()
        # Try Readability
        doc = Document(html)
        content = doc.summary()
        # Fallback: Newspaper
        if not content:
            art = Article(self.url)
            art.download(input_html=html)
            art.parse()
            content = art.text
        # Clean tags
        text = re.sub(r"<[^>]+>", "", content)
        return text


class ArxivHandler:
    def __init__(self, resource):
        self.resource = resource

    def fetch_tex(self):
        # Download source tarball via arXiv API
        arxiv_id = re.search(r"arxiv\.org/(abs|pdf)/?([^v]+)", self.resource)
        if not arxiv_id:
            raise ValueError("Invalid arXiv URL or path")
        paper_id = arxiv_id.group(2)
        url = f"https://export.arxiv.org/e-print/{paper_id}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name

    def extract_text(self):
        # Try LaTeX source first
        try:
            tar_path = self.fetch_tex()
            extract_dir = tempfile.mkdtemp()
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(extract_dir)
            # Find main .tex file
            main = None
            for root, _, files in os.walk(extract_dir):
                for fn in files:
                    if fn.endswith(".tex") and "main" in fn:
                        main = os.path.join(root, fn)
                        break
                if main:
                    break
            if not main:
                raise FileNotFoundError
            # Convert via pandoc
            cmd = ["pandoc", main, "--from=latex", "--to=plain"]
            text = subprocess.check_output(cmd, cwd=os.path.dirname(main)).decode("utf-8")
        except Exception:
            # Fallback to PDF extraction
            print("Failed to extract LaTeX return empty")
            text = ""
        return text


def process_resource(resource: str, chunk_size=1000, overlap=200):
    """
    Detects resource type, extracts and returns list of text chunks.
    """
    typ = detect_type(resource)
    if typ == "github":
        h = RepoHandler(resource)
        h.fetch()
        text = h.extract_text()
        h.clean_up()
    elif typ == "webpage":
        h = WebpageHandler(resource)
        text = h.extract_text()
    elif typ == "arxiv":
        resource = resource.replace("pdf", "html")
        result = subprocess.run(
            ["pandoc", "-f", "html", "-t", "plain", resource],
            capture_output=True,  # Captures stdout and stderr
            text=True,  # Returns stdout as str (not bytes)
            check=True,  # Raises CalledProcessError on non-zero exit
        )
        text = result.stdout

        if "HTML is not available" in text:
            h = ArxivHandler(resource)
            text = h.extract_text()
    elif typ == "kaggle":
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            text = extract_text_from_div(page, link)
            page.close()
            browser.close()
    else:
        raise ValueError(f"Unknown resource type: {typ}")

    return text


if __name__ == "__main__":
    doc = Path("challenge_doc.csv")
    doc_df = pd.read_csv(doc)
    doc_text = {}
    cache = {}

    for _, row in doc_df.iterrows():
        challenge_name = row["Challenge"]
        links = []
        for col in doc_df.columns[1:]:
            if pd.notna(row[col]):
                links.append(row[col])
        print(f"Processing {challenge_name} with links: {links}")

        link_dict = {}
        for link in links:
            try:
                if link in cache:
                    link_dict[link] = cache[link]
                else:
                    chunks = process_resource(link)
                    link_dict[link] = chunks
                    cache[link] = chunks
            except Exception as e:
                print(f"Error processing {link}: {e}")
        print("----------------------")

        doc_text[challenge_name] = link_dict

    with open("challenge_text.json", "w") as f:
        json.dump(doc_text, f, indent=4)
    print("Text extraction completed and saved to challenge_text.json")
