 Project Report: Smart Customer Grouping (K-Means Clustering)


The Problem: Treating Everyone the Same

Imagine a company has 100 customers. If they send the same coupon to everyone, they waste money on the customers who only buy cheap things, and they might miss giving a big discount to the customers who buy expensive things (their "VIPs"). They need to know who their different types of customers are.

Our AIML Goal: Automatic Grouping

Our project uses Unsupervised Machine Learning (CO4) to automatically sort all customers into 3 distinct, smart groups based on two simple numbers:

Annual Spending (How much money they spent)??

Visit Frequency (How often they shop)??

-----------------------------------------

A. The Tool

We used K-Means Clustering, which is a type of AI that finds patterns in data without needing any labels (like a teacher telling it the answers).

The K-Means algorithm runs in a loop, doing this:

Guessing: It randomly picks 3 points to be the starting centers for our 3 groups.

Measuring: It calculates the distance between every customer and these 3 centers.

Assigning: Every customer is assigned to the center they are closest to.

Moving the Center: Once all customers are assigned, the center of the group moves to the exact average position of all the customers assigned to it.

This step uses Classical Statistics (CO3) to recalculate the mean of the cluster.


-------------------------------------------------------------------------------------------------------
Technology Used

Language: Python

Core Libraries: scikit-learn (for K-Means), pandas (for data tables).

 The Result and Benefit 

The project successfully identifies three clear customer profiles, which are visible in the output graph (customer_segments.png):

Segment Label

Profile Discovered

Business Benefit

Group 0

Budget Customers: Low spending, low frequency.

Save Money: Don't waste expensive promotions on this group.

Group 1

Standard Customers: Medium spending, medium frequency.

Upsell: Focus promotions here to encourage more spending.

Group 2

VIP Customers: High spending, high frequency.

Retain: Offer exclusive perks and excellent service to keep them happy.

This AI solution replaces manual guessing with data-driven, actionable customer profiles, making marketing smarter and more efficient.
