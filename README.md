# AI OnlyFans Analytics Developer - Technical Assessment

## Business Context

Our AI chatters manage conversations with fans across multiple models. We need to understand fan behavior patterns to:

1. Train chatters on which approaches work for different fan types
2. Optimize pricing strategies per segment
3. Identify high-value fans early
4. Prevent churn before it happens

## Provided Data

- **File**: `HOMEWORK_LOGS.pkl` (5,880 messages, 100 fans, 7 days)
- **Note**: 34% of fans talk to multiple models - handle this complexity
- **Format**: Same as production (pandas DataFrame with 10 columns)
- **Size**: Intentionally minimal to protect proprietary data

## Constraints & Requirements

- **Time limit**: 1 Week from receipet
- **API budget**: Use free tiers , Voyager3 for embedding
- **Code**: Python
- **Output**: GitHub repo with reproducible results

---

## Phase 1: Data Engineering & Segmentation (25%)

### Task

Build a robust conversation segmentation pipeline that handles real-world messiness.

### Specific Requirements

1. **Handle Multi-Model Fans**:

   - 32% of fan IDs appear with multiple models
   - Treat each fan-model combination as a unique person
   - "John" on Model1 ≠ "John" on Model2
   - Each gets separate conversations, profiles, and analysis

2. **Conversation Definition**:

   - Time gap >4 hours = new conversation
   - **Model switch = ALWAYS new conversation** (critical!)
   - Each fan-model pair should be treated as a unique relationship
   - Fans with same name (e.g., "Mark") on different models are DIFFERENT fans
   - Never combine messages across different models
   - System messages should be filtered

3. **Quality Metrics**:

   - Verify no cross-model contamination in conversations

4. **Feature Engineering**:
   For each conversation, calculate these essential metrics:

   - **Revenue**: Total revenue, number of purchases, purchase rate
   - **Engagement**: Message count, duration in hours, messages before first purchase
   - **Purchase patterns**: Time to first purchase, days between first and last purchase
   - **Status**: Is customer still active? Days since last message

   These features will be used in Phase 3 clustering.

### Deliverables

- `segmentation.py` with clean, documented code
- `conversation_features.csv` with calculated metrics

---

## Phase 2: Fan Profiling with LLM (25%)

### Task

Create actionable fan profiles that chatters can actually use.

### Specific Requirements

1. **Treat each fan-model pair as unique**:

   - "Mark" on Model1 is a different person than "Mark" on Model2
   - Each fan-model pair gets its own profile
   - Focus on extracting personal details from conversations

2. **Extract Key Information**:

   - Demographics (age indicators, relationship status)
   - Personal details (job, hobbies, location hints)
   - Life events (divorced, new job, health issues)
   - Personality traits and emotional needs
   - What motivates their purchases
   - Communication patterns

3. **API Optimization**:
   - Batch 20 conversations per API call
   - Implement caching to avoid duplicate calls
   - Handle API failures gracefully
   - Track costs: Log tokens used

### Deliverables

- `fan_profiler.py` with retry logic
- Sample profiles showing extracted personal details
- Documentation of your profiling approach

---

## Phase 3: Conversation Stage Analysis & Clustering (25%)

### Task

Segment each fan-model relationship into stages, create embeddings, and find patterns through clustering.

### Specific Requirements

1. **Stage Segmentation**:
   For each fan-model pair, split their entire conversation history into 3 stages:

   - **Stage 1**: First message → First sale (or end if no sale)
   - **Stage 2**: First sale → Last sale (skip if only one sale)
   - **Stage 3**: Last sale → Last message (or current if still active)

   Each stage contains ALL messages in that time period.

2. **Create Embeddings - TWO WAYS**:

   **Method A - Without Fan Profiling**:

   - Concatenate all messages in each stage
   - Create embeddings using Voyage-3 or OpenAI
   - This gives you pure conversation patterns

   **Method B - With Fan Profiling**:

   - Combine message embeddings with fan profile features
   - Include extracted details (job, hobbies, personality traits)
   - This tests if profiling improves clustering

3. **Clustering & Visualization**:

   - Apply k-means clustering (test different k values)
   - Create 2D visualizations (t-SNE or UMAP)
   - Create 3D interactive plots
   - Compare clusters from Method A vs Method B
   - Color code by revenue, retention, or other metrics

4. **Cluster Assessment**:
   - What patterns emerge in each stage?
   - Which clusters are most valuable?
   - Do profile-enhanced embeddings create better clusters?
   - What actionable insights can you extract?

### Deliverables

- `stage_segmentation.py` - Code to split conversations
- `embeddings_without_profile.pkl` - Method A embeddings
- `embeddings_with_profile.pkl` - Method B embeddings
- `clustering_analysis.py` - Your clustering code
- `visualizations/` - 2D and 3D plots
- `cluster_insights.md` - Your findings and recommendations

---

## Phase 4: Actionable Insights & Chatter Playbook (25%)

### Task

Transform your clustering analysis into practical strategies that chatters can use.

### Specific Requirements

1. **Cluster Interpretation**:

   - Give each cluster a memorable name
   - Describe the typical fan in each cluster
   - What makes each cluster unique?
   - Which clusters are most/least valuable?

2. **Chatter Strategies**:
   For each cluster, provide specific guidance:

   - Best times to engage
   - Conversation style that works
   - Optimal price points
   - What triggers purchases
   - Red flags to watch for

3. **Business Recommendations**:

   - Which clusters should we focus on?
   - How can we move fans from low to high-value clusters?
   - What experiments should we run?
   - Expected revenue impact of your recommendations

4. **Stage-Specific Insights**:
   - What works in acquisition (Stage 1)?
   - What maximizes revenue (Stage 2)?
   - What prevents churn (Stage 3)?

### Deliverables

- `chatter_playbook.md` - Practical guide for each cluster
- `business_recommendations.md` - Strategic insights
- `experiment_proposals.md` - Specific A/B tests to run
- Executive presentation (slides or document)

---

## Bonus Challenges (+20% each, max 2)

### 1. Churn Prediction

- Identify fans likely to stop responding
- Provide intervention strategies
- Baseline: 70%+ accuracy on 30-day churn

### 2. Optimal Message Timing

- When should chatters message each fan?
- Consider timezone, historical response rates
- Baseline: 20%+ improvement in response rate

### 3. Multi-Model Strategy

- Which fans should be introduced to other models?
- Cross-sell opportunity sizing

---

## Submission Instructions

### GitHub Repository Structure

Organize your code in a clear, logical structure with:

- Source code files
- Documentation
- Output visualizations
- At least one test file
- Clear README with setup instructions

### Executive Summary Requirements

1. **Business Impact** (1 paragraph)

   - Revenue opportunity identified
   - Cost savings from better targeting
   - Efficiency gains for chatters

2. **Technical Approach** (1 paragraph)

   - Key innovations in your solution
   - Trade-offs made and why

3. **Next Steps** (3-5 bullets)
   - How to productionize
   - Additional data needed
   - Recommended A/B tests

---

## Evaluation Rubric

### Code Quality (30%)

- Clean, modular, documented code
- Proper error handling
- Performance optimization
- Tests for critical functions

### Business Value (30%)

- Actionable insights
- Clear ROI calculations
- Practical recommendations
- Chatter-friendly outputs

### Technical Depth (25%)

- Handling edge cases
- Creative feature engineering
- Appropriate algorithm choices
- Scalability considerations

### Communication (15%)

- Clear documentation
- Professional visualizations
- Executive-friendly summary
- Code readability

---

## FAQ

**Q: The data has fans talking to multiple models, but instructions assumed single model?**
A: Each fan-model pair should be analyzed separately. Fans with the same name on different models are different people. Never mix messages across models.

**Q: What if my clusters don't match the baseline metrics?**
A: Explain why. Maybe you found better patterns. We value reasoning over matching numbers.

**Q: Can I use additional data sources?**
A: No external data, but you can engineer any features from what's provided.

**Q: What if I run out of API credits?**
A: Process a subset and note scalability approach. We care more about method than completeness.

---

## Final Notes

- We use this exact analytics pipeline in production
- The best submission will be invited to present findings to our team
- Strong candidates understand both the technical and business challenges
- Show personality in your solution - we want to see how you think

Good luck! Reach out if you have clarifying questions in the first 24 hours.

---

## Advanced Bonus Challenge (Choose ONE for +30% score)

### Option A: Semantic Search with Vector DB (Tests: Vector DB, RAG basics)

Build a system where chatters can search for similar conversations:

- Store conversation embeddings in a vector database (Pinecone free tier, Chroma, or pgvector)
- Implement semantic search: "Find fans similar to this one"
- Create a simple RAG system: Given a fan message, retrieve 5 similar past conversations and suggest responses
- Deliverable: `semantic_search.py` with API endpoint

**What we're looking for:**

- Proper vector indexing strategy
- Metadata filtering (by model, success rate)
- Understanding of similarity vs keyword search

### Option B: Purchase Prediction Model (Tests: Supervised ML)

Build a model to predict purchase probability:

- Use first N messages as features
- Create labels from whether they eventually purchased
- Compare multiple approaches (logistic regression, XGBoost, simple NN)
- Feature engineering is key (message sentiment, time patterns, etc.)
- Deliverable: `purchase_predictor.py` with 80%+ AUC

**What we're looking for:**

- Proper train/test splitting (time-based!)
- Feature engineering creativity
- Model explainability for chatters

### Option C: Fine-tuned Response Generator (Tests: LLM fine-tuning)

Fine-tune a small model for better chatter responses:

- Use successful conversations as training data
- Fine-tune a small model (GPT-3.5, Llama2-7B, or similar)
- Compare responses to base model
- Focus on model safety (no inappropriate responses)
- Deliverable: Model weights + evaluation notebook

**What we're looking for:**

- Data quality for training
- Evaluation methodology
- Safety considerations
- Cost/benefit analysis

**Evaluation Criteria for Advanced Bonus:**

- Technical implementation (10%)
- Business value demonstrated (10%)
- Production readiness (10%)
- Choose based on your strengths - we value depth over breadth
