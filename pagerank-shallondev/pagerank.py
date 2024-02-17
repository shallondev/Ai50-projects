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
    
    # Find linked pages and number of total pages
    linked_pages = corpus[page]
    total_pages = len(corpus)
    total_linked_pages = len(linked_pages)

    # Handle case when there are no linked pages
    if total_linked_pages == 0: 
        linked_pages = corpus.keys() 
        total_linked_pages = total_pages
    
    # Calculate probabilites for choosing linked and unlinked pages
    probability_linked = damping_factor / total_linked_pages 
    probability_unlinked = (1 - damping_factor) / total_pages

    # Calculate probability distributions for unlinked and linked pages
    probability_distribution = {
        page: probability_unlinked for page in corpus
    }
    for linked_page in linked_pages:
        probability_distribution[linked_page] += probability_linked

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create a list of all pages
    pages = list(corpus.keys())

    # Select first sample at random
    sample = random.choice(pages)

    # Create counter to count occurance of each sample
    sample_count = {sample: 1}

    # Iterate over remaining samples
    for _ in range(n-1):

        # Get probability distribution from transition model
        probability_distribution = transition_model(corpus, sample, damping_factor)

        # Get probabilities associated with 
        probabilities = list(probability_distribution.values())

        # Choose a new sample based on transition_model
        sample = random.choices(pages, probabilities, k=1)[0]

        # Increment count for the chosen sample
        sample_count[sample] = sample_count.get(sample, 0) + 1

    # Build a page ranking distribution based on our sampling
    page_rank_distribution = {
        page: count / n for page, count in sample_count.items()
    }

    return page_rank_distribution


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Variables
    N = len(corpus)
    d = damping_factor

    # Initialize PageRank values
    pagerank = {page: 1 / N for page in corpus}

    # First part of the formula
    pr_general = (1 - d) / N

    # Repeat until all page values change by less than 0.001
    while True:

        # Initialze new page ranks dict
        new_pagerank = {}

        # Iterate all pages in corpus to apply formula
        for page in corpus:

            # Initialize second part of formula
            pr_link = 0

            # Consider each page in corpus
            for i in corpus:

                # Special case when page links to nothing
                if len(corpus[i]) == 0:
                    pr_link += d * pagerank[i] / len(corpus.keys())

                # Case when page links to us
                elif page in corpus[i]:
                    pr_link += d * pagerank[i] / len(corpus[i])

            # Calculate the new rank of the page
            new_pagerank[page] = pr_general + pr_link

        # Check for convergence
        if all(abs(pagerank[page] - new_pagerank[page]) < 0.001 for page in corpus):
            pagerank = new_pagerank
            break

        # Otherwise update page rank and repeat
        pagerank = new_pagerank

    # Return page rank
    return pagerank


if __name__ == "__main__":
    main()
