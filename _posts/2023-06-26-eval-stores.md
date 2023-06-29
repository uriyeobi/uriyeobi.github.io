---
layout: post
title: "Evaluation Stores - a high bias, low variance view"
# use_math: true
# comments: true
tags: 'MLOps'
# excerpt_separator: <!--more-->
# sticky: true
# hidden: true
---


[Feature Store](https://www.featurestore.org/what-is-a-feature-store) has been one of the hottest buzzwords in the machine learning community in recent years. In my view, however, **"Evaluation Store"** should be of equal or higher priority in many teams. In this post, I will write about the main idea of evaluation stores and explain why I think it is critical.

---


<br>

# Example


Suppose you are building ML solutions for a certain loan business.

You developed a ML model that predicts default probability for each borrower. 

And you deployed the model.


## Checking model performance

A few weeks later, you wondered how it has been going, so checked the model performance, by doing the following tasks:


- Load the model (object) in production.
- Load the loan-level data, macroeconomic data, and other third party data.
- Merged them to create a target modeling dataset.
- Run the model to the dataset, to produce predicted default rates for each loan.
- Aggregate the actual and predicted default rates per time period (say, 2 weeks).
- Calculate error metrics.

Here is a diagram of the above tasks.

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/8191304dd60b895754f9266ba49c8ee15e7307ca/assets/images/posts_eval_stores/eval_store_tasks.png?raw=true" width="800rem">


## Poor performance detected
The output from the above tasks looks like this:

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/8191304dd60b895754f9266ba49c8ee15e7307ca/assets/images/posts_eval_stores/eval_store_diagnosis_fig1.png?raw=true" width="700rem">

Suppose you found that the model performed very poorly for the period `2023-04-10 ~ 2023-04-23`.

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/8191304dd60b895754f9266ba49c8ee15e7307ca/assets/images/posts_eval_stores/eval_store_diagnosis_fig2.png?raw=true" width="700rem">

## Looking deeply
You wanted to understand why, so you looked into it with more granular views like state-level. For that, you conducted the similar tasks as above, but now with an additional slicing - state.

<br>

Now, within the period `2023-04-01 ~ 2023-04-23`, it is revealed that the model performed especially bad for the borrowers who live in `NY`:

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/8191304dd60b895754f9266ba49c8ee15e7307ca/assets/images/posts_eval_stores/eval_store_diagnosis_fig3.png?raw=true" width="700rem">


<br>

## Looking more deeply
You are still not sure about why, so you dived into this sliced data by further slicing it by occupation, and discovered that the model performed the worst for the group of borrowers with `Science & Engineering` occupation.

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/8191304dd60b895754f9266ba49c8ee15e7307ca/assets/images/posts_eval_stores/eval_store_diagnosis_fig4.png?raw=true" width="800rem">


Why does the model perform badly for this slice of data (`2023-04-10 ~ 2023-04-03` + `NY` + `Science & Engineering`? At this point, you may or may not find the reason. You may need to try other slices. You may also need to dig into macro-economic data, third-party data, bugs in codes, bugs in model applications, etc. You would continue until you get some clues.



# Problems

The above example is not unusual - many data scientists actually do this kind of work. Is there anything wrong here?

> Nothing is wrong, if this happens only once, for a single model, and for a single data scientist. But that's not the case mostly.

If you have a ML team, imagine how many times and how many data scientists have to do similar things repeatedly over time. Not only that, but the following questions can arise:


- Data Slices
	- How do you pull each data? What is the version of each?
	- How do you join the multiple datasets?
	- What is the definition of each data slicing?
- Metrics 
	- How do you aggregate loan-level quantities to group-level quantities?
	- What is the (mathematical) definition of each metric?
	- How can the metrics calculation codes be shared with others?
	- How do you decide whether calculated performance metrics are acceptable or not?
	- How do you relate the model metrics to business metrics?
- Multiple Models: 
	- How do you compare multiple models effectively?
	- How do you select the model?
- Monitoring and Diagnosis
	- How do you deliver insights about the model performance to internal or external groups?
	- What is an efficient playbook to find root causes of model degradation?

So your team needs to discuss how to answer these questions. Note that none of these are about developing sophisticated machine learning models, real-time training or rigorous deployment, etc. All we want here is simple - to understand how the model performs in some data slices. But as we saw from the example, evaluating models involves various components with details, which makes it nontrivial in the end. 
As a result, *doing such analytics may eat up all of your time*. 

So efficiency and transparency are needed here.

# Evaluation Stores

An evaluation store is a single place where model performances are summarized for different data slices or use cases.

It can play a critical role for the team. For example, it is obvious that monitoring and reporting can be benefited from evaluation stores. A new ideation to improve the current model should also start from the current model performance analysis. Or, when you talk with other team members about model issues, you have to refer to a single source of truth for the metrics. 


<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/8191304dd60b895754f9266ba49c8ee15e7307ca/assets/images/posts_eval_stores/eval_store_overview.png?raw=true" width="700rem">

So key benefits of evaluation stores are:
- Reducing redundancy model analytics queries/scripts
- Enhancing transparency for metrics calculations
- Accelerating model experiments[^2]
- Identifying poor-performing data and model segments quicker
- Unifying the model reporting channels across the teams


# Designing Evaluation Stores

There is no standard rule or format on how to build an evaluation store. It all depends on teams and businesses. Building a scalable evaluation store actually could be an overkill in many small companies. So each team should develop its own process taking into account the core ideas of evaluation stores. 

Here are some possible designs of evaluation stores that I thought of:

### Version 0
- Create a dashboard for the production model performance
- Share frequently used scripts or queries
	
### Version 1
- Incorporate different data and model versions
- Design data slicing in multi-scale
- Centralize metrics implementation
- Create evaluation tables whose fields include:
	- Model versions
	- Data versions
	- Data slices
	- Metrics

### Version 2
- Cover both offline and online model evaluation metrics coherently and dynamically update them
- Devise an automatic alert engine
- Devise a smart root cause-finding algorithm (for poorly performed model/data slices)
- Incorporate features evaluations
- Incorporate business metrics evaluations
- May be built on the top of feature store and [model registry](https://neptune.ai/blog/ml-model-registry)

### Version 3
- More advanced functionalities..



# Concluding Remarks

There could be a lot of frustration that comes with evaluating models.

Some of them actually can be resolved with feature stores. A feature store enables reusing features effectively across the team and facilitates various tasks in ML pipelines. But it is not easy to build an infrastructure for that. Plus, the feature drifts are not that common and small errors in features do not necessarily cause severe impacts on the product. 

On the other hand, **model performances are directly linked to the product's success**. So the process of diagnosing the model performances must be innovated. Data scientists should start working on models, with much better understanding of model performance with almost no costs.

In terms of building an evaluation store, I would prefer a lean approach - starting from building a minimal version, and make it evolve with the progression of teams, products and infrastructures. 


    
# References
  
* Lecture 10 of Full Stack Deep Learning Course, Spring 2021: [link](https://fullstackdeeplearning.com/spring2021/lecture-10/)
* Talk by Josh Tobin, Gantry - Feature Stores and Evaluation Stores: Better Together [link](https://youtu.be/mOAfRtFh7zw)
* ML systems design course at Stanford - CS 329S: Machine Learning Systems Design: [link](https://stanford-cs329s.github.io/syllabus.html)
* Paper on slicing - Slice finder: Automated data slicing for model validation, Y. Chung, et al., 2019 IEEE 35th International Conference on Data Engineering (ICDE): [link](https://research.google/pubs/pub47966/)
* Medium article: Evaluation Stores: Closing the ML Data Flywheel?: [link](https://farmi.medium.com/evaluation-stores-closing-the-ml-data-schwungrad-b2429cc80981)
* Blog posts from neptune.ai:
	- The Ultimate Guide to Evaluation and Selection of Models in Machine Learning: [link](https://neptune.ai/blog/ml-model-evaluation-and-selection)
	- ML Model Registry: What It Is, Why It Matters, How to Implement It: [link](https://neptune.ai/blog/ml-model-registry)
	- Feature Stores: Components of a Data Science Factory: [link](https://neptune.ai/blog/feature-stores-components-of-a-data-science-factory-guide)
* Feature Store Summit 2023: [link](https://www.featurestoresummit.com)

<br>


****


**Notes**

[^2]: By model experiments, I meant the iterative process of ideation, building new features, tuning hyperparameters, training, validation, backtesting, and model decisions 

