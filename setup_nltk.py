import nltk

PACKAGES = [
    "wordnet",
    "omw-1.4",
    "stopwords",
    "averaged_perceptron_tagger_eng",
]


def main() -> None:
    for pkg in PACKAGES:
        print(f"Downloading NLTK package: {pkg}...")
        nltk.download(pkg, quiet=False)
    print("All NLTK packages downloaded successfully.")


if __name__ == "__main__":
    main()
