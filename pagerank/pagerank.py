import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probs = {page:0 for page in corpus} #probability is initally 0
    if len(corpus[page]) == 0:
        probs = {page:1/len(corpus.keys()) for page in corpus} #if the page has no links, then a new page is selected randomly, meaning all chances are equal
        return probs
    rP = (1-damping_factor)/len(corpus[page]) #probability from randomly being chosen
    dP = damping_factor/len(corpus) #probability from getting clicked on in a link in hte current webpage
    for p in probs:
        probs[p]+=rP #adds the chance of being randomly chosen
        if p in corpus[page]: #if the page is linked in the current page...
            probs[p] += dP #add the chance of being clicked on in the page
    return probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {c:0 for c in corpus.keys()} #holds count of how many times each page is visited (dividing by n gives the chance)

    page = random.choice(tuple(corpus.keys())) #randomly chooses an initial page
    ranks[page]+=1 #increases the count of the initial page
    for i in range(n-1): #gets n-1 samples (subtracts one because the first page is the first sample)
        num = random.random() #random number
        chances = transition_model(corpus, page, damping_factor) #gets the chances
        c=0
        for ii in chances: #chooses a random webpage based on the given chances
            c+=chances[ii]
            if num<=c:
                page=ii
                ranks[page]+=1
                break
    return {c[0]: c[1] / n for c in ranks.items()} #returns the rankings

def iterate_pagerank(corpus, damping_factor):
    """
        Return PageRank values for each page by iteratively updating
        PageRank values until convergence.

        Return a dictionary where keys are page names, and values are
        their estimated PageRank value (a value between 0 and 1). All
        PageRank values should sum to 1.
    """
    ranks = {page:1/len(corpus) for page in corpus}
    QUIT_THRESHOLD = 0.001 #if the amount of changes is really low ,then stop the thing
    changes = {p:1 for p in corpus} #keeps track of the changes in each page each iteration
    while max(changes.values())>QUIT_THRESHOLD:
        for page in ranks: #constantly iterates through pages
            prevval = ranks[page]
            linkingPages = set()
            for p in corpus:
                if page in corpus[p]:
                    linkingPages.add(p)
            sumlinks = 0
            for p in linkingPages:
                sumlinks+=ranks[p]/len(corpus[p])
            ranks[page] = ((1-damping_factor)/len(corpus))+damping_factor*sumlinks #sets the new rank based on some math stuff
            changes[page] = abs(prevval-ranks[page])


    return ranks


if __name__ == "__main__":
    main()
