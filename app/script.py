import pandas as pd

def main():
    df = pd.read_csv("data/data.csv")  # data.csv ada di host, dimount ke /app/data
    print("Baris:", len(df))
    print(df.head())

if __name__ == "__main__":
    main()