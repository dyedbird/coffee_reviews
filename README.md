# coffee_reviews
The data set was pulled from kaggle  via: https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset
The features are as follows:

    name: Name of the blend
    roaster: Name of the roaster
    roast: Type of roast (Light, Medium-Light, Medium, Medium-Dark, Dark)
    loc_country: Location of the roaster
    origin_1: Origin of the beans
    origin_2: Origin of the beans
    100g_USD: price per 100g of beans in US dolalrs
    rating: Rating of the coffee
    review_date: Date of the coffee review
    review: Text of review #1 + Text of review #2 + Text of review #3

  After cleaning and vectorizing the text data, hierarchical agglomerative clustering is used to group the reviews via text similarity.
  Resulting clusters are then visualized using NetworkX and Plotly with a slider feature for similarity score values. The slider allows for a min similarity score to be used as a filter for visualization.
  Subsequently, corpora specific to each cluster is fed into a multi index data frame.
  An ensemble topic model (LDA) is then run on each cluster corpus where it is automated to optimize for greatest coherence score.
  Resulting keywords for each topic / cluster are plotted on a collective horizontal bar chart where bar colors are automated to coordiante with the colors from the cluster plot.
  The work can be expanded to generate a fully fledged recommendation engine based on the liked / dislike coffee beans


![newplot](https://github.com/user-attachments/assets/00a13928-8343-4672-9e46-5c6e5b145530)

![Cluster_Topics_Orig](https://github.com/user-attachments/assets/1a42c496-056f-47d8-8080-194c50f509c4)
