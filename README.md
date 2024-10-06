# Recommendation System:
**Recommendation system** is an algorithm that suggests items users may like based on their preferences, behaviors, and interactions. There are three main types:

1. **Content-Based Filtering:** Recommends items similar to those a user has liked before, based on item features (e.g., genre, actors).
   
2. **Collaborative Filtering:** Suggests items based on user similarity, recommending items liked by similar users.

3. **Hybrid Systems:** Combine content-based and collaborative filtering for more accurate recommendations.

These systems are used in various platforms like e-commerce, streaming services, and social media to personalize user experiences and increase engagement.<br>
Here the recommendation system is build for movies and It is of **Hybrid type(both content and collaborative filtering is used)**.<br>
# Dataset:
Movielens Dataset of 9000 movies is used. Dataset can be found [here](https://grouplens.org/datasets/movielens/)
# Methodology:
Both Content Based filtering and Collaborative filtering is used. short description of them is given below:<br>
## Content Based filtering:
Content-Based Filtering is a recommendation approach that suggests items to users based on the features of the
items and the user's past preferences. It operates under the principle that if a user liked an item, they will also like similar items that share comparable characteristics.<br>
**Mathematical Formula:**
Given:
- A user $$\( u \)$$
- An item $$\( i \)$$
- Features of items represented as vectors in a feature space

The similarity between user preferences $$\( P_u \)$$ (a vector of features representing the user’s interests) and item features $$\( F_i \)$$ (a vector of features representing the item) can be calculated using the **cosine similarity** formula:

$$\[
\text{Similarity}(u, i) = \frac{P_u \cdot F_i}{\|P_u\| \|F_i\|}
\]$$

Where:
- $$\( P_u \cdot F_i \)$$ is the dot product of the user profile and item feature vectors.
- $$\( \|P_u\| \) and \( \|F_i\| \)$$ are the magnitudes (or norms) of the user and item vectors, respectively.

The system recommends items with the highest similarity scores to the user based on their profile.<br>
## Collaborative filtering with Singular Value Decomposition (SVD):
**Collaborative Filtering** is a method used to recommend items to users based on the preferences of similar users. One popular approach to collaborative filtering is using **Singular Value Decomposition (SVD)**, which reduces the dimensionality of the user-item interaction matrix to capture latent features.

#### Key Steps in Collaborative Filtering with SVD:

1. **Matrix Factorization:**  
   Given a user-item interaction matrix \( R \) where rows represent users and columns represent items, SVD decomposes \( R \) into three matrices:
   
   
   $$R \approx U \Sigma V^T$$
   
   - $$\( U \)$$: User feature matrix (size $$\( m \times k \)$$)
   - $$\( \Sigma \$$): Diagonal matrix of singular values (size $$\( k \times k \)$$)
   - $$\( V^T \)$$: Item feature matrix (size $$\( k \times n \)$$)
   - $$\( m \)$$: Number of users, $$\( n \)$$: Number of items, $$\( k \)$$: Number of latent features

2. **Prediction:**  
   The predicted rating $$\( \hat{R}_{ui} \)$$ for user $$\( u \)$$ on item $$\( i \)$$ can be calculated as:
   
   
   $$\hat{R}_{ui} = U_u \cdot \Sigma \cdot V_i^T$$
   
   where $$\( U_u \)$$ is the user vector for user $$\( u \) and \( V_i \)$$ is the item vector for item $$\( i \)$$.

#### Loss Function

To optimize the SVD model, we define a loss function that captures the difference between the predicted and actual ratings. A common choice is the **Mean Squared Error (MSE)** loss function:


$$L = \sum_{(u,i) \in K} (R_{ui} - \hat{R}_{ui})^2$$


where $$\( K \)$$ is the set of user-item pairs with known ratings.

#### Update Rule

In the context of gradient descent optimization for matrix factorization, the update rules for the user and item matrices are as follows:

1. **User and Item Updates:**  
   For a given user $$\( u \) and item \( i \)$$ with learning rate $$\( \alpha \)$$:

   
   $$U_u \leftarrow U_u + \alpha \cdot \left( (R_{ui} - \hat{R}_{ui}) \cdot V_i - \lambda U_u \right)$$
   

   
   $$V_i \leftarrow V_i + \alpha \cdot \left( (R_{ui} - \hat{R}_{ui}) \cdot U_u - \lambda V_i \right)$$
   

   Where:
   - $$\( \lambda \)$$: Regularization parameter to prevent overfitting.
   - $$\( R_{ui} \)$$: Actual rating for user $$\( u \) on item \( i \)$$.
   - $$\( \hat{R}_{ui} \)$$: Predicted rating for user $$\( u \) on item \( i \)$$.

### Conclusion

By iteratively updating the user and item matrices based on the gradient of the loss function, SVD allows the model to learn latent features that
effectively capture user preferences and item characteristics, ultimately leading to better recommendations.<br>
# Results:
run hybrid_recommend.ipynb file, in the last cell replace Roll Bounce with the movie of your choice in *my_favorite_movies = ['Roll Bounce']*. 10 recommendations each based on similar movies and users will be shown in output.<br>
result for Roll Bounce is shown below:<br>
![download](https://github.com/SameerSri72/Movie_recommendation_system-hybrid-/blob/main/pic2.png)<br>
# Application Development:
A Simple Application is built using streamlit and successfully tested on local system. deployment is not done..( certain version conflicts need to be resolved, hopefully it will be updated soon).<br>
Screenshot of application is shown below:<br>
![download](https://github.com/SameerSri72/Movie_recommendation_system-hybrid-/blob/main/pic.png)<br>
# References:
1. SVD-based incremental approaches for recommender systems (Xun Zhou et. al) [link](https://www.sciencedirect.com/science/article/pii/S0022000014001706)<br>
2. [Cory Maklin](https://medium.com/@corymaklin/model-based-collaborative-filtering-svd-19859c764cee)<br>
3. [Analytics Dimag](https://analyticsindiamag.com/ai-mysteries/singular-value-decomposition-svd-application-recommender-system/)<br>



