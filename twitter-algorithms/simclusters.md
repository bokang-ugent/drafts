#### 概览
* SimClusters作为描述用户兴趣的稀疏嵌入特征，是推特推荐链路在召回阶段最重要的向量特征之一[1]。Twitter在2020年发表的相关文章中提到，研发的SimClusters的动机主要是为了向多个下游任务提供通用的特征空间，进而以此提升这些任务的精度[2]。其主要思路是基于推特关注网络对影响力高的用户做社区发现，将得到的社区作为用户兴趣的嵌入特征。用户以外的其他推特实体（例如，推文，话题，趋势等）也可以通过聚合与其交互过的用户兴趣特征得到。这篇文章还提到，在上线SimCluster特征后，多个下游任务的表现都有了不同程度的提升。
* 推特在2020年发表SimClusters文章时，同时开源了一个基于java的代码实现[3]。在2023年的三月，SimClusters的Scala实现也随着推特的推荐算法的公开而开源了[4]。在最近的版本中，simclusters_v2文件夹下的README.md[5]对SimClusters的计算过程进行了更直观的总结。这些代码，文档以及论文对SimClusters在不同的侧面进行了描述，因此对照研究可以增进对算法的理解。
* 本文以近期公开的simcluster文档中的概念和符号表示为基础，将SimClusters描述的的核心计算流程和Scala代码进行关联和解读。希望以此能为对SimClusters感兴趣的读者提供一些思路。

#### 概念和符号表示[5, 2]
* $A_{m\times n}$ - 推特关注用户二部网络的邻接矩阵。其中 $m$ 个用户 被划分到内容的消费侧（consumers）, $n$ 个用户被划分到内容的生产侧（producers）。推特的关注网络大约有$10^{11}$条边，$10^7$个生产侧用户，$10^9$个消费侧用户。划分依据？
* $\{\tilde{A}^T\tilde{A}\}_{n\times n}$ - 生产侧用户的相似度矩阵。该矩阵为计算余弦相似度然后进行阈值过滤所得。其中 $\tilde{A}_{m\times n}$ 为对二部图邻接矩阵中每个生产者列归一化后的矩阵，即$\tilde{A}_{:,j} = A_{:,j}/||A_{:,j}||$。该相似度矩阵大约有$10^7$个用户，$10^9$条边。
*  $V_{n\times k}$ - "大V"用户矩阵（Known for）。该矩阵作为后续其他嵌入矩阵（embedding matrix）的计算基础，包含了$n$个生产侧用户的嵌入向量，向量的每一个维度代表一个兴趣社区，一共$k$个兴趣社区。每一个社区都代表了一个兴趣相似的消费者群体，或者代表了一个生产者被周知（known for）的领域。该矩阵为对生产侧用户的余弦相似度矩阵应用SimClusters文章中新提出的社区发现算法得到。为了保持稀疏性，每一个生产者用户最多有一个相关的社区。在推特的数据中，一共有大约$10^5$个社区。
* $U_{m\times k}$  - 消费侧用户兴趣矩阵（Consumer Embeddings - User InterestedIn）。该矩阵刻画消费侧用户的兴趣，由关注矩阵和Known for矩阵相乘所得，$U_{m\times k} = A_{m\times n } \cdot V_{n\times k}$。为了节约存储空间，过滤掉较小的元素，每个用户只保留少量非零元素（推特应用中大约10~100）。
*  $\tilde{V}_{n\times k}$ - 生产侧用户兴趣矩阵（Producer Embedding)。由于Known For矩阵$V$过于稀疏，不能很好刻画生产侧用户作为内容消费者的兴趣[5]，需要进一步的计算余弦相似度：$\tilde{V}_{n\times k} = \tilde{A}^T\cdot \tilde{U}$, 其中$\tilde{A}$ 为对每个producer维度归一化的矩阵$\tilde{A}_{:,i} = A_{:,i}/||A_{:,i}||$, $\tilde{U}$ 为对每一个兴趣社区维度归一化的矩阵$\tilde{U}_{:,j} = U_{:,j}/||U_{:,j}||$。
* $W_{t\times k}$ - 其他twitter实体矩阵（Entity Embeddings）。其他实体包含推文（tweet），话题（topic），趋势（trend）。该矩阵为与实体进行交互（e.g., 点赞）的用户兴趣向量进行聚合所得。聚合函数为按交互发生时间指数衰减的函数。该聚合函数的优点在于可以对 $W$ 矩阵进行在线增量更新。

#### 核心计算流程和关联代码
* 第一阶段：用户社区发现
	* 第一步: 计算生产侧用户相似度矩阵 $\{\tilde{A}^T\tilde{A}\}_{n\times n}$
		* 以推特的用户数量级，生产侧用户相似度矩阵的计算非常具有挑战性。而文章[2]中提到由于该计算问题在产品中有非常重要的应用，推特用了大量的资源研发了一个可靠的算法，即WHIMP 算法。WHIMP算法是一种应用Wedge sampling 和 Locality sensitive hashing 对用户相似度进行模拟计算的sketch算法。算法细节已发表在WWW 2017[6]这里不做赘述。
	* 第二步: 生产侧用户社区发现，计算$V_{n\times k}$ 矩阵（Algorithm 1 - 3, [2]）
		* 为了有效的表示生产侧用户相似图的结构，社区需要将大小控制在数百个用户，而不是更大的数量级[2]。对于$10^7$的生产侧用户，其潜在的社区数量大致为$10^5$ 。而针对这种规模，在SimClusters论文发表的当时没有现成算法可以应用。于是推特提出了Neighborhood-Aware-Metropolis–Hastings算法。其中Metropolis–Hastings为常见的蒙特卡洛采样算法。在社区发现的上下文里，其主要思路是随机的采样$k$维二值向量，生成用户兴趣社区的随机备选方案。然后对采样的方案用目标函数进行打分。如果一个备选方案有更高的目标函数值则被接受为当前方案，否则按一定概率拒绝。
		* 目标函数相似图中的邻居用户之间的社区重合度，反之降低相似度不高的用户之间的社区重合度。进一步，对第一项进行加权是$Z$在优化过程中被分配更多的非$0$项.
			* $f(u, Z)\triangleq\alpha\sum_{v\in\mathcal{N}(u)}\mathbb{1}(|Z(u)\cap Z(v)|>0) + \sum_{v\notin\mathcal{N}(u)}\mathbb{1}(|Z(u)\cap Z(v)|=0)$
			* 可以观察到，优化目标函数的第一项将相似图中临近的用户聚集在一起，类比于对比学习中的pulling force；同样，第二项类比于pushing force。因此对比学习的一些方法（例如，负采样）也应该可以在这里使用。
		* 为了进一步加速算法，SimClusters文章中还提到由于一个用户属于其邻居所属的社区集合以外社区的可能性极低，因此只用从邻居所属兴趣社区集合中采样子集作为备选方案即可，这也是名称中的Neighborhood-Aware的由来。在枚举子集，并且计算目标函数的时候，可以通过复用计算过的交集加快计算速度。在计算接受概率时也一样。softmax采样也可以采用Gumbel-Max trick加速。
		* 限制每一个用户最多只属于一个社区，可以进一步降低每一个epoch的复杂度
		* 因为目标函数可分解，即可通过对每个节点分别计算然后相加得到。可以用过同步并行，甚至异步并行来有效的进行计算。
		* 代码实现
			* Batch任务 - 主要描述在 `src/scala/com/twitter/simclusters_v2/scalding/update_known_for/` 文件夹 `UpdateKnownFor20M145K2020.scala` 文件中定义。[第97行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownFor20M145K2020.scala#L97)通过调用 `UpdateKnownForSBFRunner.runUpdateKnownFor(...)` 进入社区发现流程。
			* 社区发现流程 - 于`UpdateKnownForSBFRunner.scala`的[第53行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L53) `runUpdateKnownFor(...)` 函数中定义。
				* [第74行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L74) 加载最有影响力的20M的用户关注图
				* [第82行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L82) 加载消费侧用户相似度图 `SimsGraph`
				* [第658行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L658) 加载之前计算的Known for矩阵来初始化社区发现算法的变量。如果用户以前没有被分到社区，则随机的选择一些未使用的社区。
				* [第668行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L668) `getClusteringAssignments()`函数中， 调用 [第536行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L536) 的`optimizeSparseBinaryMatrix()` 函数，然后调用了Neighborhood-Aware-Metropolis–Hastings算法的`optimize()`函数。
				* [第132行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/update_known_for/UpdateKnownForSBFRunner.scala#L132) 输出计算结果作为新的Known for矩阵。
			* Neighborhood-Aware-MH算法 - 主要位于`twitter/sbf` 这个git repo下面 [MHAlgorithm.java](https://github.com/twitter/sbf/blob/master/src/main/java/com/twitter/sbf/core/MHAlgorithm.java) 文件中。
				* [第102行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L102) `optimize()` - 进入优化流程，定义优化的主循环。每个循环调用`runEpoch()`函数
				* [第279行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L279) `runEpoch()` - 进入每个epoch的计算流程
					* [第296行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L296) 串行计算模式
					* [第330行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L330) 并行带同步锁的模式
					* [第345行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L345) 并行允许异步更新的模式，与HogWild![7]文章中提出的ASGD(Asynchronous Stochastic Gradient Descent)方法的思路类似。
				* [第839行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L839) `mhStep()`函数以上的三种模式的每个epoch计算中都被调用，对应文章[2]中的Algorithm 1和Algorithm 3
					* [第843行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L843) 按照不同的模式生成候选社区分配
					* [第860行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L860) 计算候选分配的目标函数值
					* [第864行](https://github.com/twitter/sbf/blob/41afeaeb6de374dd0cb30aaa9ba6cd618e546de6/src/main/java/com/twitter/sbf/core/MHAlgorithm.java#L864) 根据目标函数值按概率决定接收或拒绝当前候选方案
	* 第三步: 计算消费侧用户兴趣矩阵 $U_{m\times k}$  (Section 3.3, [2])
		* 由关注矩阵和Known for矩阵相乘所得，$U_{m\times k} = A_{m\times n } \cdot V_{n\times k}$。为了存储的考虑，每个用户只保留了固定的非零项。
		* 代码实现主要在 `simclusters_v2/scalding/InterestedInFromKnownFor.scala` 文件下
			* [第54行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L54) 进入主要计算流程
			* [第78行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L78) 加载user-user 关注 + 点赞 图
			* [第80行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L80) 加载Know for嵌入向量
			* [第570行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L570) `run()` 调用  [第530行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L570) `keepOnlyTopClusters()` 调用  [第342行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L342) `attachNormalizedScores()` 调用  [第249行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/InterestedInFromKnownFor.scala#L249)  `userClusterPairsWithoutNormalization()` 用矩阵乘法计算余弦相似度矩阵，消费者侧用户兴趣矩阵。
* 第二阶段：其他推特实体的嵌入(Algorithm 4, [2])
	* 这一阶段的计算以用户-实体交互矩阵和用户兴趣矩阵为输入。每一实体的嵌入向量可以通过对于该实体交互过的用户兴趣向量进行聚合得到。聚合函数为按交互发生时间指数衰减的函数。
	* 按照时效性，嵌入计算又分为离线和在线模式。生产侧用户兴趣嵌入，话题嵌入为离线批量计算。推文嵌入和趋势嵌入为在线计算。
	* 代码实现
		* 离线
			* `simclusters_v2/scalding/embedding/ProducerEmbeddingsFromInterestedIn.scala`
				* [第539行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/ProducerEmbeddingsFromInterestedIn.scala#L539) ProducerEmbeddingsFromInterestedIn -> runOnDateRange -> getProducerClusterEmbedding
				* [第592行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/ProducerEmbeddingsFromInterestedIn.scala#L592) getProducerClusterEmbedding -> SimClustersEmbeddingJob.legacyMultiplyMatrices
			* `simclusters_v2/scalding/embedding/EntityToSimClustersEmbeddingsJob.scala`
				* name with "Adhoc" runs the adhoc job for quick prototyping, name with out Adhoc are for production
				* [第143行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/EntityToSimClustersEmbeddingsJob.scala#L143) EntityToSimClustersEmbeddingApp -> [第238行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/EntityToSimClustersEmbeddingsJob.scala#L238) computeEmbeddings()
				* `embedding/common/SimClustersEmbeddingJob.scala] [第94] computeEmbeddings() -> multiplyMatrices
			* `simclusters_v2/scalding/embedding/tfg/FavTfgBasedTopicEmbeddings.scala`
				* Fav-based Topic-Follow-Graph (TFG)
				* what is "LogFavBasedTweet"? -> seems to be log transformation, search "val logFavScore =" and see "logTransformation".
				* [第58行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/tfg/FavTfgBasedTopicEmbeddings.scala#L58) FavTfgTopicEmbeddingsScheduledApp -> TfgBasedTopicEmbeddingsBaseApp
				* `TfgBasedTopicEmbeddingsBaseApp.scala` [第37行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/tfg/TfgBasedTopicEmbeddingsBaseApp.scala#L37) TfgBasedTopicEmbeddingsBaseApp -> SimClustersEmbeddingBaseJob
				* `embedding/common/SimClustersEmbeddingJob.scala` [第94行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scalding/embedding/common/SimClustersEmbeddingJob.scala#L94) SimClustersEmbeddingBaseJob -> computeEmbeddings() -> multiplyMatrices 
					``` scala
					/**
					* This is the base job for computing SimClusters Embedding for any Noun Type on Twitter, such as
					* Users, Tweets, Topics, Entities, Channels, etc.
					*
					* The most straightforward way to understand the SimClusters Embeddings for a Noun is that it is
					* a weighted sum of SimClusters InterestedIn vectors from users who are interested in the Noun.
					* So for a noun type, you only need to define `prepareNounToUserMatrix` to pass in a matrix which
					* represents how much each user is interested in this noun.
					*/
					trait SimClustersEmbeddingBaseJob[NounType]
					```
			* [GCP] `simclusters_v2/scio/bq_generation`
				* "We have a GCP pipeline where we build our SimClusters ANN index via BigQuery. This allows us to do fast iterations and build new embeddings more efficiently compared to Scalding."
				* PushOpenBased SimClusters ANN Index: The job builds a clusterId -> TopTweet index based on user-open engagement history. This SANN source is used for candidate generation for Notifications.
				* VideoViewBased SimClusters Index: The job builds a clusterId -> TopTweet index based on the user's video view history. This SANN source is used for video recommendation on Home.
				* [第42行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scio/bq_generation/simclusters_index_generation/EngagementEventBasedClusterToTweetIndexGenerationJob.scala#L42) EngagementEventBasedClusterToTweetIndexGenerationJob
					* [第90](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scio/bq_generation/simclusters_index_generation/EngagementEventBasedClusterToTweetIndexGenerationJob.scala#L90) getTopKTweetsForClusterKeyBQ: Generate SimClusters cluster-to-tweet index via BQ
					* [第125行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scio/bq_generation/simclusters_index_generation/EngagementEventBasedClusterToTweetIndexGenerationJob.scala#L125) [第137行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/scio/bq_generation/simclusters_index_generation/EngagementEventBasedClusterToTweetIndexGenerationJob.scala#L137) Save SimClusters index to a BQ table and KeyValSnapshotDataset
		* 在线
			* `simclusters_v2/summingbird/storm/TweetJob.scala`
				* [第90行](https://github.com/twitter/the-algorithm/blob/6e5c875a69b5dc400302e42a3d0b2cfe509c71b6/src/scala/com/twitter/simclusters_v2/summingbird/storm/TweetJob.scala#L90) SimClustersInterestedInUtil.buildClusterWithScores
				* `summingbird/common/SimClustersInterestedInUtil.scala` buildClusterWithScores -> ThriftDecayedValueMonoid -> `ThriftDecayedValueMonoid.scala` -> [com.twitter.algebird.DecayedValueMonoid.scala](https://twitter.github.io/algebird/datatypes/decayed_value.html)
			* `simclusters_v2/summingbird/storm/PersistentTweetJob.scala`
				``` scala
				/**
				* The job to save the qualified tweet SimClustersEmbedding into Strato Store(Back by Manhattan).
				*
				* The steps
				* 1. Read from Favorite Stream.
				* 2. Join with Tweet Status Count Service.
				* 3. Filter out the tweets whose favorite count < 8.
				* We consider these tweets' SimClusters embedding is too noisy and untrustable.
				* 4. Update the SimClusters Tweet embedding with timestamp 0L.
				* 0L is reserved for the latest tweet embedding. It's also used to maintain the tweet count.
				* 5. If the SimClusters Tweet embedding's update count is 2 power N & N >= 3.
				* Persistent the embeddings with the timestamp as part of the LK.
				**/
				```
#### Deployment 
* Representations are keyed by model version. This allows parallel evaluation alongside production.
* Training and operating 
	* Stage 1
		* step 1 most expensive (2days), was in production before. User-user does not vary much in size
		* step 2 run once and update to reflect user-user graph change
		* step 3 periodically run
	* Stage 2
		* [batch] user influence representation. In terms of user influencing in communities. $10^8$ user with 10 - 100 non-zeros
		* [batch] topic representation. In terms of community interested in a topic.
		* [stream] tweet representation. user-tweet engagements based, ~$10^2$ non-zeros
		* [stream] trend representation. user-tweet engagements based, ~$10^2$ non-zeros
		* Inverted top-k indices (users/topics/tweets/trends for community)

#### Applications / AB Test
* Similar tweets
	* previously: solely based on author similarity
	* now: (main tweet, tweet) similarity (25% higher engagement rate), (user-influence of main tweet, tweet) similarity (7% extra improvement)
* Tweet recommendation on home page
	* previously: "Network activity" algorithm - GraphJet based
	* now: supplement (tweets, user-influence) similarity, （candidate tweets, recently-liked tweets). 33% module improve, ~1% total improve
	* extra: use user interests, and tweet representations as feature to improve ranking (i.e., engagement prediction). 4.7$ improvement.
* Ranking of personalized trends
	* a two-stage process: trend detection, ranking
	* previously: volume based + small number of personalization features
	* new: (user interest, trend representation) 8% increase within trend, 12% increase on landing page subsequent to a click.
* Topic recommendation
	* predefined topic taxonomy
	* previously: text matching rules curated by human experts -> false positives
	* now: (tweet, topic) similarity + rule
* Who to follow
	* previously: an engagement prediction model
	* now: add (user, user) similarity as feature, 7% increase
* In progress
	* Notification quality filter. User-user blocking graph, use representation as feature to train model to filter out abusive or spamy replies or mentions. 4% lift PR-AUC
	* Supervised embeddings: add other user/feature (in addition to various engagement graphs), e.g., address, follower accounts, train prediction task with two-tower model. 
	* Real-time event notification: event representation - aggregated from human-created tweits about the event -> identify community would be interested -> target users

#### Next
* Python implementation + simulation?

#### References
[1] Twitter's Recommendation Algorithm, Twitter Engineering Blog, 2023
[2] Wu et al., SimClusters: Community-Based Representations for Heterogeneous Recommendations at Twitter, KDD, 2023
[3] https://github.com/twitter/sbf
[4] https://github.com/twitter/the-algorithm
[5] https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/simclusters_v2/README.md
[6] Sharma et al., When Hashes Met Wedges: A Distributed Algorithm for Finding High Similarity, WWW, 2017
[7] Recht et al., Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent, NeurIPS, 2011
