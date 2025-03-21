# 🏡 Property Match Finder

## **Overview**
Property Match Finder is a machine-learning-powered recommendation system that helps users find the best property match based on their preferences. Using numerical and textual data, we compute a **Match Score** for each property to help users make informed decisions.

## **Table of Contents**
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Features](#features)
- [Installation & Setup](#installation--setup)
---

## **Problem Statement**
Users looking for a property often have multiple preferences, such as budget, number of bedrooms, and qualitative descriptions. However, finding an **optimal match** among several properties is challenging.  

### **Key Challenges:**
- Users struggle to find properties that match both their **quantitative** (budget, bedrooms) and **qualitative** (text descriptions) preferences.  
- Existing filtering mechanisms do not consider **semantic understanding** of property descriptions.  
- Need for a **simple UI** to display and interpret match scores effectively.  

---

## **Solution Approach**
Developed a **machine-learning-based recommendation system** that:
- Uses **numerical feature scaling** to standardize property and user preferences.
- Leverages **Natural Language Processing (NLP)** to compare qualitative descriptions.
- Combines **quantitative and qualitative similarity** into a **Match Score** for each property.
- Provides an **interactive UI** using Gradio for easy selection and interpretation of results.

---

## **Dataset**
The system operates on an Excel dataset with two sheets:  
1️⃣ **User Data**: Contains preferences such as budget, number of bedrooms, bathrooms, and qualitative descriptions.  
2️⃣ **Property Data**: Contains available properties with similar attributes.  


## **Methodology**
The **Match Score** is computed using a combination of **numerical similarity** and **text similarity**:

### **Step 1: Data Preprocessing**
- Extract numerical values from budget and price.
- Normalize numerical features (Budget, Bedrooms, Bathrooms) using **MinMax Scaling**.

### **Step 2: Compute Numeric Similarity**
- Calculate Euclidean distance between user preferences and property attributes.
- Convert distance to a similarity score.

### **Step 3: Compute Text Similarity**
- Convert qualitative descriptions into vector embeddings using **SentenceTransformers**.
- Compute **Cosine Similarity** between user and property descriptions.

### **Step 4: Final Match Score**
The final score is a **weighted average** of numeric and text similarity:
\[
\text{Match Score} = 0.5 \times \text{Numeric Similarity} + 0.5 \times \text{Text Similarity}
\]
---

## **Features**
✅ **Machine Learning-Powered Recommendation** – Uses NLP and numerical features.  
✅ **Gradio UI** – Simple dropdown-based property selection.  
✅ **Match Score Interpretation** – Provides explanations for score values.  
✅ **Efficient Computation** – Scales and processes data dynamically.  

---

## **Installation & Setup**
### **Prerequisites**
- Python 3.8+
- Install dependencies:
  ```bash
  pip install pandas numpy gradio scikit-learn sentence-transformers openpyxl
