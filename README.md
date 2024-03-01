# RagAgents

<img src="https://files.oaiusercontent.com/file-BQWpIKuXxgcDuK3RY1KNPRuT?se=2024-03-01T10%3A47%3A01Z&sp=r&sv=2021-08-06&sr=b&rscc=max-age%3D31536000%2C%20immutable&rscd=attachment%3B%20filename%3D703cf855-3ee1-4ec5-8a17-c152cb50b1d2.webp&sig=1kV3XcXrxaoIwAJEaJUaxI/EHrg9t1yif7UsnUT1Sik%3D" style="zoom: 33%;" />

## Introduction

In the evolving landscape of software engineering, language models have carved a niche for themselves, offering unparalleled advantages in planning and reasoning. Yet, these models are not without their flaws. The emergence of hallucination problems within language models can disrupt logical planning, introduce uncertainties, and hinder the execution of complex tasks. Our project, at the intersection of database knowledge and advanced language models, introduces a Retrieval Augmentation Generation (RAG) System. This system is tailored for the DBA Agency, designed to navigate through the intricacies of task planning with precision and efficiency.

### The Core Challenges

Our journey through developing this RAG system has led us to confront several intricate challenges:

1. **Navigating Comprehensive DBA Documentation**: The vast and multimodal nature of DBA O&M documentation presents a formidable challenge, requiring a sophisticated path to knowledge extraction.
2. **Intelligent Knowledge Retrieval**: Crafting a system capable of understanding user intent and retrieving pertinent database knowledge from a sea of information demands multi-source, multi-mode, and multi-round capabilities.
3. **Synthesis of Fragmented Information**: The art of combining disparate pieces of retrieved information into a coherent whole is a puzzle we aim to solve.
4. **Efficient Task Planning and Reporting**: Our goal is to leverage the synthesized information to plan tasks and generate comprehensive reports.
5. **Integration of Diverse Components**: The project tackles the integration of intent comprehension, knowledge retrieval, information synthesis, and task planning into a cohesive system.
6. **Evaluating System Effectiveness**: Developing a robust framework for testing and evaluating the system's effectiveness remains a paramount objective.

### Evaluating Our Progress

The efficacy of our RAG system is measured through a multifaceted evaluation approach:

- **Intent Comprehension**: We assess the accuracy and completeness of the system's understanding of user intent.
- **Retrieval Module**: This module's performance is gauged through metrics like Hit Rate, Mean Reverse Ranking (MRR), Normalised Discounted Cumulative Gain (NDCG), and Precision.
- **Generation Module**: Here, we focus on the contextual relevance of the combined documents and queries.
- **Task Planning Module**: The evaluation revolves around the correctness, relevance, and executability of the generated task plans.

An **end-to-end assessment** further complements our evaluation methods, offering a holistic view of the system's response to user inputs across all modules.
