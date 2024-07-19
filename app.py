from flask import Flask, request, render_template, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

app = Flask(__name__)
warnings.filterwarnings("ignore")

# Path to the Chrome WebDriver executable
webdriver_path = "C:\\Windows\\chromedriver-win64\\chromedriver.exe"

# Load RoBERTa tokenizer and model for sentiment analysis
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize logs variable
logs = []

# Function to clean and tokenize text and compute sentiment scores


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Function to clean the review text


def clean_text(text):
    # Remove newlines and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to scrape Amazon reviews


def scrape_amazon_reviews(url):
    global logs
    logs = []

    # Configure Chrome options
    chrome_options = Options()
    # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--headless")

    # Initialize Chrome WebDriver
    service = Service(webdriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Open the URL
    driver.get(url)

    try:
        # Wait for review bodies to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'span[data-hook="review-body"] span')
            )
        )
        # Find all review texts
        review_elements = driver.find_elements(
            By.CSS_SELECTOR, 'span[data-hook="review-body"] span'
        )
        review_texts = [clean_text(review.text) for review in review_elements]

        # Create a DataFrame
        data = {'Review Text': review_texts}
        df = pd.DataFrame(data)

    except Exception as e:
        logs.append(f"An error occurred: {str(e)}")
        df = pd.DataFrame()

    finally:
        # Close the WebDriver
        driver.quit()

    return df

# Function to perform sentiment analysis


def perform_sentiment_analysis(df):
    res = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Review Text']
            roberta_result = polarity_scores_roberta(text)
            res[i] = roberta_result
            logs.append(f'Successfully analyzed review {i}')
        except RuntimeError:
            logs.append(f'Error analyzing review {i}')

    # Convert results to DataFrame
    res_df = pd.DataFrame.from_dict(res, orient='index')
    df = df.join(res_df)

    return df

# Function to plot sentiment scores for individual reviews


def plot_individual_sentiment_scores(df):
    images = []
    filtered_df = df.dropna(
        subset=['roberta_neg', 'roberta_neu', 'roberta_pos'])
    for i, row in filtered_df.iterrows():
        plt.figure(figsize=(6, 4))
        plt.bar(['Negative', 'Neutral', 'Positive'], [row['roberta_neg'],
                row['roberta_neu'], row['roberta_pos']], color=['red', 'blue', 'green'])
        plt.xlabel('Sentiment')
        plt.ylabel('Score')
        plt.title(f'Sentiment Scores for Review {i+1}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        images.append(image_base64)
        plt.close()
    return images


def plot_pairplot(df):
    filtered_df = df.dropna(
        subset=['roberta_neg', 'roberta_neu', 'roberta_pos'])
    plt.figure(figsize=(10, 6))
    pairplot = sns.pairplot(
        filtered_df[['roberta_neg', 'roberta_neu', 'roberta_pos']])
    buf = io.BytesIO()
    pairplot.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def plot_histogram(df):
    filtered_df = df.dropna(
        subset=['roberta_neg', 'roberta_neu', 'roberta_pos'])
    plt.figure(figsize=(10, 6))
    plt.hist([filtered_df['roberta_neg'], filtered_df['roberta_neu'], filtered_df['roberta_pos']],
             bins=20, label=['Negative', 'Neutral', 'Positive'], color=['red', 'blue', 'green'])
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sentiment Scores')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def plot_boxplot(df):
    filtered_df = df.dropna(
        subset=['roberta_neg', 'roberta_neu', 'roberta_pos'])
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=filtered_df[['roberta_neg', 'roberta_neu', 'roberta_pos']], palette=[
                'red', 'blue', 'green'])
    plt.xlabel('Sentiment')
    plt.ylabel('Score')
    plt.title('Boxplot of Sentiment Scores')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def plot_pie_chart(df):
    filtered_df = df.dropna(
        subset=['roberta_neg', 'roberta_neu', 'roberta_pos'])
    sentiment_counts = [
        (filtered_df['roberta_neg'] > 0.5).sum(),
        (filtered_df['roberta_neu'] > 0.5).sum(),
        (filtered_df['roberta_pos'] > 0.5).sum()
    ]
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=[
            'Negative', 'Neutral', 'Positive'], autopct='%1.1f%%', colors=['red', 'blue', 'green'])
    plt.title('Pie Chart of Sentiment Distribution')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def plot_line_plot(df):
    filtered_df = df.dropna(
        subset=['roberta_neg', 'roberta_neu', 'roberta_pos'])
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df.index,
             filtered_df['roberta_neg'], label='Negative', color='red')
    plt.plot(filtered_df.index,
             filtered_df['roberta_neu'], label='Neutral', color='blue')
    plt.plot(filtered_df.index,
             filtered_df['roberta_pos'], label='Positive', color='green')
    plt.xlabel('Review Index')
    plt.ylabel('Score')
    plt.title('Line Plot of Sentiment Scores Over Reviews')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


@app.route('/', methods=['GET', 'POST'])
def index():
    global logs
    if request.method == 'POST':
        url = request.form['url']
        df = scrape_amazon_reviews(url)
        df = perform_sentiment_analysis(df)

        # Generate individual plots
        images = plot_individual_sentiment_scores(df)

        # Generate pairplot
        pairplot_image = plot_pairplot(df)

        # Generate histogram
        histogram_image = plot_histogram(df)

        # Generate boxplot
        boxplot_image = plot_boxplot(df)

        # Generate pie chart
        piechart_image = plot_pie_chart(df)

        # Generate line plot
        lineplot_image = plot_line_plot(df)

        # Convert DataFrame to HTML
        df_html = df.to_html(classes='table table-striped', index=False)

        return render_template('index.html', tables=df_html, images=images, pairplot_image=pairplot_image,
                               histogram_image=histogram_image, boxplot_image=boxplot_image,
                               piechart_image=piechart_image, lineplot_image=lineplot_image)

    return render_template('index.html')


@app.route('/logs', methods=['GET'])
def get_logs():
    global logs
    return jsonify(logs)


if __name__ == '__main__':
    app.run(debug=True)
