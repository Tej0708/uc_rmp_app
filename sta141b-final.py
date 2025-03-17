import streamlit as st
import pandas as pd
import plotly.express as px
import ast
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter


df = pd.read_csv("uc_rmp_professors_full.csv")
df_all = pd.read_csv("uc_rmp_reviews.csv")

## CLEAN AND FORMAT DATA
df = df.drop(['Mode Rating'], axis=1)
df.columns = df.columns.str.replace(" ", "_")

df["Would_Take_Again"] = df["Would_Take_Again"].str.replace("%", "", regex=False)
df["Would_Take_Again"] = pd.to_numeric(df["Would_Take_Again"], errors="coerce")

if isinstance(df_all["Sentiment"].iloc[0], str):  # Check if the first row is a string
    df_all["Sentiment"] = df_all["Sentiment"].apply(ast.literal_eval)




department_df = df.groupby(['Department']).mean(numeric_only=True)
#st.dataframe(department_df)

def metric_by_dept(selected_dept, metric):
    # Filter for the selected department
    filtered_df = df[df["Department"] == selected_dept]
    
    grouped_df = filtered_df.groupby("School").agg({
        metric: "mean",
        "Num_Ratings": "sum"
    }).reset_index()

    fig = px.bar(
        grouped_df,
        x="School",
        y=metric,
        text="Num_Ratings",  # This column will be used to display text on the bars
        title=f"Average metrics by School for the {selected_dept} Department",
        labels={"School": "UC Campus", metric: f"Mean of {metric}"}
    )

    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},  
        template="plotly_white"
    )
    
    st.plotly_chart(fig)
    st.write("The numbers inside the bars are the number of ratings this school had for that metric")

def sentiment_by_university():
    df_sentiment = df_all.explode("Sentiment")

    # Check if the DataFrame is empty
    if df_sentiment.empty:
        st.warning("No sentiment data available for visualization.")
        return None
    df_sentiment_count = df_sentiment.groupby(["School", "Sentiment"]).size().reset_index(name="Count")

    # Create interactive stacked bar chart
    fig = px.bar(
        df_sentiment_count, 
        x="School", 
        y="Count", 
        color="Sentiment", 
        barmode="stack", 
        title="Sentiment Breakdown per University",
        labels={"Count": "Number of Reviews", "School": "University", "Sentiment": "Sentiment Type"},
        color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"}
    )

    fig.update_layout(
        width=1000,  
        height=600,  
        xaxis_tickangle=-45  # Rotate x-axis labels
    )

    return fig
def plot_sentiment_counts():
    df_sentiment_counts = df_all.explode("Sentiment")["Sentiment"].value_counts()

    # Create figure and axis
    fig, ax = plt.subplots()
    df_sentiment_counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)

    # Set labels and title
    ax.set_title("Sentiment Distribution of Reviews")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_xticklabels(df_sentiment_counts.index, rotation=0)

    # Show plot in Streamlit
    st.pyplot(fig)


def plot_wordcloud():
    #nltk.download("stopwords") RUN THIS CODE IF YOU DON'T HAVE STOPWORDS DOWNLOADED
    stop_words = set(stopwords.words("english"))
    custom_stop_words = {'class', 'lecture', 'professor', 'really', 'homework', 'take', 'final', 'lectures', 'get', 'exams', 'student', 
                        'midterm', 'students', 'stats', 'go', 'make', 'course', 'pretty', 'would', 'like', 'much' }
    stop_words.update(custom_stop_words)

    # Combine all reviews into a single list
    all_reviews = " ".join(df_all["Reviews"].explode().astype(str)).lower()
    words = [word for word in all_reviews.split() if word not in stop_words and word.isalpha()]

    # Count word frequencies
    word_freq = Counter(words)
    df_words = pd.DataFrame(word_freq.most_common(20), columns=["Word", "Count"])
        # Word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Most Frequent Words in Reviews")
    st.pyplot(fig)

def plot_sentiment_with_rating():
    def categorize_rating(rating):
        if rating >= 4.5:
            return "High Rating"
        elif rating >= 3.0:
            return "Medium Rating"
        else:
            return "Low Rating"

    # Apply the rating category based on the Average Rating
    df_all['Rating_Category'] = df_all['Average Rating'].astype(float).apply(categorize_rating)

    # Exploding the Sentiment column to have individual rows for each sentiment
    df_exploded = df_all.explode('Sentiment')

    # Count the sentiment distribution across different rating categories
    sentiment_counts = df_exploded.groupby(['Rating_Category', 'Sentiment']).size().reset_index(name='Count')

    # Plot the sentiment distribution across different rating categories
    fig = px.bar(sentiment_counts, x='Rating_Category', y='Count', color='Sentiment',
                title="Sentiment Distribution Based on Rating Category",
                labels={"Rating_Category": "Rating Category", "Sentiment": "Sentiment"},
                color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig)
def plot_sentiment_ratio_with_rating():
    # Function to calculate Positive/Negative Sentiment Ratio
    def sentiment_ratio(sentiments):
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        if negative_count == 0:
            return positive_count  # To avoid division by zero, return the positive count if no negative sentiments
        return positive_count / negative_count

    # Apply the function to calculate the ratio for each professor
    df_all['Pos_Neg_Ratio'] = df_all['Sentiment'].apply(sentiment_ratio)
    df_all['Average Rating'] = df_all['Average Rating'].astype(float)

    # Create scatter plot: Average Rating vs Positive/Negative Sentiment Ratio
    fig = px.scatter(df_all, x='Average Rating', y='Pos_Neg_Ratio', 
                    title="Average Rating vs Positive/Negative Sentiment Ratio",
                    labels={"Average Rating": "Average Rating", "Pos_Neg_Ratio": "Positive/Negative Sentiment Ratio"},
                    color = 'School',
                    hover_data=['Professor'],
                    trendline="ols",  # Ordinary least squares regression (line of best fit)
                    trendline_scope="overall")  # Adding professor as hover data for context
    st.plotly_chart(fig)



## PLOTTING CODE
# Sidebar Menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis"])

if page == 'Home':
    st.title("Analysis of Student Sentiment of Professors at the University of California system")
    st.write("This app will give insights into how students feel about their professors across many different academic departments for all 9 UCs")
    st.subheader("Here is the data we are working with:")
    with st.expander("Data"):
        st.dataframe(df)

    with st.expander('Metrics by Department across the UCs'):
        departments = df["Department"].unique()
        selected_dept = st.selectbox("Select a Department", options = sorted(departments))
        selected_metric = st.selectbox("Select a Metric", options=["Average_Rating", "Level_of_Difficulty", "Would_Take_Again"])
        metric_by_dept(selected_dept, selected_metric)

    with st.expander("Professor Data by School"):
        st.write("Filter by your desired school and see what people are saying about professors here!")

        school_options = sorted(df["School"].unique())
        selected_school = st.selectbox("Select a School", options=school_options)

        school_df = df[df["School"] == selected_school]

        dept_breakdown = school_df.groupby("Department").agg({
            "Average_Rating": "mean",
            "Level_of_Difficulty": "mean",
            "Would_Take_Again": "mean",
            "Num_Ratings": "sum"
        }).reset_index()

        st.subheader(f"Department Breakdown for {selected_school}")
        st.dataframe(dept_breakdown)

        fig = px.bar(
            dept_breakdown,
            x="Department",
            y="Average_Rating",
            title=f"Average Rating by Department for {selected_school}",
            labels={"Department": "Department", "Average_Rating": "Mean Rating"}
        )
        fig.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
        st.plotly_chart(fig)
if page == 'Sentiment Analysis':
    st.title("Sentiment Analysis")
    st.write("We wanted to specifically look at the reviews students left and conduct some sentiment analysis. Since each professor has hundreds of reviews, it would take a lot of time to scrape every single department. Therefore we decided to scrape only the Statistics department.")
    st.write("Due to the long runtime, we only scraped the Statistics Department of UCSD, UC Davis, UC Berkeley, UCLA, and UCI.")
    st.subheader("Here is the data we are working with:")
    with st.expander("Data"):
        st.dataframe(df_all)
    st.write("For the sentiment analysis, one of the main goals was to understand how the different universities' statistics departments differed in terms of their professors. This is why the first charts we plotted where to see how many reviews were of each sentiment and how they were based on each university.")
    plot_sentiment_counts()
    st.write("The first chart we plotted was sentiment distribution. Overall, there were way more positive reviews than negative, basically more than double, so we could possibly infer the that the statistics departments generally have good professors overall. However, to specifically narrow down the distribution of reviews over different universities, we created a different plot based on the university.")
    st.plotly_chart(sentiment_by_university())
    st.write("The first thing that you can notice, is that UC Davis seems to have the most amount of professors in the statistics department, compared to all the other schools. Interestingly enough, all schools had a higher proportion of professors with positive reviews vs. negative reviews. This tells us that students of each university look upon their statistics department in a favorable light. Next we wanted to see what are some common words that are used in reviews. For this reason we created a word cloud. We made sure to remove the regular stop words, and other common words that did not provide value like “lecture” or “class”.")
    plot_wordcloud()
    st.write("For some context, the larger the word on the wordcloud, the more frequently it appears. The most frequently used words include adjectives like good, hard, and easy. Something interesting from this chart would be that the size of both hard and easy is relatively the same. Based on the previous chart, we say that there were way more positive reviews than negative reviews. However, the frequence of the words hard and easy are roughly the same. This means that there could be phrases in reviews like “hard but fair”, which are actually positive reviews.")
    plot_sentiment_with_rating()
    st.write("Next we wanted to see the distribution of positive to negative reviews based on the professors average rating. Since students can rank the rating of a professor and write a review, we wanted to see if those had some sort of correlation with each other. We had to create a qualitative variable for the average rating and so we said that anything above a 4.5 was a high rating, between a 3 and 4.5 was a medium rating, and below 3 was a low rating. As we would think there were mostly positive reviews for high rating professors, however, what was interesting was that low rating professors had roughly similar positive review numbers to negative review numbers. This probably means that lower ratings don’t necessarily mean a really bad professor all time, it could also be a professor that has mixed reviews. Finally we wanted to see exactly how these two are related. So we created another column in our dataset which corresponded to the positive/negative review ratio. We then plotted the average rating for a professor vs this ratio.")
    plot_sentiment_ratio_with_rating()
    st.write("In this plot, we also wanted to see if the university had any effect on the points which we can clearly see it did not. We also fit a simple linear regression model to see if there was any relationship between the average rating and the positive/negative review ratio. Although it looks like there is a small positive correlation between the two variables, the R2 of 0.211 indicates that average rating doesn’t account for much of the variation of the positive/negative review ratio. This means that rating and the postive/negative reviews might actually not have a relationship and that although we might think of these as two correlated metrics, they in fact can differ.")
    
