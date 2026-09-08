
import os
import time
import requests
import json
from supabase import create_client
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import feedparser

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStorage:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.model = SentenceTransformer(MODEL_NAME)
    def embed_text(self, text):
        return self.model.encode(text).astype("float32").tolist()
    def chunk_text(self, content, chunk_size=500):
        words = content.split()
        chunks = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > chunk_size and current:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks
    def store_document(self, name, format, content):
        doc_res = self.client.table("coreknow_documents").insert({"name":name,"format":format,"content":content}).execute()
        doc_id = doc_res.data[0]["id"]
        for chunk in self.chunk_text(content):
            emb = self.embed_text(chunk)
            self.client.table("coreknow_chunks").insert({"document_id":doc_id,"chunk_text":chunk,"embedding":emb}).execute()
        return doc_id

class MultiSourceSearch:
    def __init__(self):
        self.ua = "CoreKnow/1.0"
        self.wikipedia = "https://en.wikipedia.org/w/api.php"
        self.arxiv = "http://export.arxiv.org/api/query"
        self.openalex = "https://api.openalex.org/works"
        self.crossref = "https://api.crossref.org/works"
        self.semantic = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.pubmed = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.wikidata = "https://www.wikidata.org/w/api.php"
        self.github = "https://api.github.com/search/repositories"

    def search_wikipedia(self, q, limit=3):
        try:
            params = {"action":"query","list":"search","srsearch":q,"srlimit":limit,"format":"json"}
            r = requests.get(self.wikipedia, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                return [{"title":x["title"],"snippet":x.get("snippet",""),"source":"Wikipedia"} for x in r.json().get("query",{}).get("search",[])]
        except: pass
        return []

    def search_duckduckgo(self, q, limit=3):
        try:
            r = requests.post("https://html.duckduckgo.com/html/", data={"q":q}, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                soup = BeautifulSoup(r.text,"html.parser")
                return [{"title":a.get_text(),"url":a.get("href",""),"source":"DuckDuckGo"} for a in soup.find_all("a",class_="result__a")[:limit]]
        except: pass
        return []

    def search_arxiv(self, q, limit=3):
        try:
            params = {"search_query":f"all:{q}","start":0,"max_results":limit}
            r = requests.get(self.arxiv, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                feed = feedparser.parse(r.text)
                return [{"title":e.title,"summary":e.summary,"link":e.link,"source":"ArXiv"} for e in feed.entries[:limit]]
        except: pass
        return []

    def search_openalex(self, q, limit=3):
        try:
            params = {"search":q,"per-page":limit}
            r = requests.get(self.openalex, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                return [{"title":w.get("title",""),"doi":w.get("doi",""),"source":"OpenAlex"} for w in r.json().get("results",[])[:limit]]
        except: pass
        return []

    def search_crossref(self, q, limit=3):
        try:
            params = {"query":q,"rows":limit}
            r = requests.get(self.crossref, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                items = r.json().get("message",{}).get("items",[])
                return [{"title":item.get("title",[""])[0] if item.get("title") else "","DOI":item.get("DOI",""),"source":"Crossref"} for item in items[:limit]]
        except: pass
        return []

    def search_semantic(self, q, limit=3):
        try:
            params = {"query":q,"limit":limit,"fields":"title,abstract,url"}
            r = requests.get(self.semantic, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                return [{"title":p.get("title",""),"abstract":p.get("abstract",""),"url":p.get("url",""),"source":"Semantic Scholar"} for p in r.json().get("data",[])[:limit]]
        except: pass
        return []

    def search_pubmed(self, q, limit=3):
        try:
            params = {"db":"pubmed","term":q,"retmax":limit,"retmode":"json"}
            r = requests.get(self.pubmed, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                return r.json().get("esearchresult",{}).get("idlist",[])
        except: pass
        return []

    def search_wikidata(self, q, limit=3):
        try:
            params = {"action":"wbsearchentities","search":q,"language":"en","limit":limit,"format":"json"}
            r = requests.get(self.wikidata, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                return [{"id":x["id"],"label":x.get("label",""),"description":x.get("description",""),"source":"Wikidata"} for x in r.json().get("search",[])[:limit]]
        except: pass
        return []

    def search_github(self, q, limit=3):
        try:
            params = {"q":q,"per_page":limit}
            r = requests.get(self.github, params=params, headers={"User-Agent":self.ua}, timeout=15)
            if r.status_code==200:
                return [{"name":i.get("full_name",""),"description":i.get("description",""),"url":i.get("html_url",""),"source":"GitHub"} for i in r.json().get("items",[])[:limit]]
        except: pass
        return []

    def deep_search(self, q):
        combined = f"Deep search: {q}\n\n"
        for source, items in {
            "wikipedia": self.search_wikipedia(q),
            "duckduckgo": self.search_duckduckgo(q),
            "arxiv": self.search_arxiv(q),
            "openalex": self.search_openalex(q),
            "crossref": self.search_crossref(q),
            "semantic_scholar": self.search_semantic(q),
            "pubmed_ids": self.search_pubmed(q),
            "wikidata": self.search_wikidata(q),
            "github": self.search_github(q)
        }.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        combined += str(item.get('title') or item.get('label') or item.get('name','')) + " "
                        combined += str(item.get('snippet') or item.get('summary') or item.get('abstract') or item.get('description','')) + "\n"
                    else:
                        combined += str(item) + "\n"
            else:
                combined += str(items) + "\n"
        return combined

class ContinuousLearner:
    def __init__(self):
        self.vs = VectorStorage()
        self.search = MultiSourceSearch()
        self.queue = self.load_topics()
        self.visited = set()

    def load_topics(self):
        try:
            with open("topics.json") as f:
                data = json.load(f)
                topics = []
                for cat in data.values():
                    topics.extend(cat)
                return topics[:500]
        except:
            return ["physics","mathematics","chemistry","biology","artificial intelligence"]

    def find_related(self, topic):
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {"action":"query","prop":"links","titles":topic,"pllimit":10,"format":"json"}
            r = requests.get(url, params=params, timeout=15, headers={"User-Agent":"CoreKnow/1.0"})
            if r.status_code == 200:
                pages = r.json().get("query",{}).get("pages",{})
                for pid in pages:
                    return [l["title"] for l in pages[pid].get("links",[])[:10]]
        except: pass
        return []

    def learn_topic(self, topic):
        if topic in self.visited:
            return
        self.visited.add(topic)
        content = self.search.deep_search(topic)
        if content:
            self.vs.store_document(topic, "deep_search", content)
            print(f"[LEARNED] {topic}")
            related = self.find_related(topic)
            for r in related:
                if r not in self.visited and r not in self.queue:
                    self.queue.append(r)
        else:
            print(f"[NO DATA] {topic}")

    def run(self):
        print("CoreKnow 24/7 worker started.")
        while self.queue:
            topic = self.queue.pop(0)
            self.learn_topic(topic)
            time.sleep(3)
        print("Queue empty. Worker exiting.")

if __name__ == "__main__":
    ContinuousLearner().run()
