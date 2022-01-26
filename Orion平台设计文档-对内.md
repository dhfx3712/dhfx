

[TOC]



# Orion 平台设计文档

## 平台简介

`Orion` 自动化机器学习平台是一个简单易操作、统一接口的机器学习平台，管理机器学习任务的全流程，包括以下几个步骤：

* 数据预处理
* 特征工程
* 模型训练
* 模型管理
* 模型预测
* 用户交互

目前的自动化机器学习平台支持分类、回归、聚类、时间序列分析四个建模场景，具备一个完整的平台型技术能力，可以为金融、制造、零售等领域的业务数据进行快速建模和分析。

## 平台架构

### 总体架构

自动化机器学习平台的[总体架构](https://drive.google.com/file/d/1a6KhJuHtvUubDbIB9Gis1Tq1OTrtgsYD/view)如下图所示。

![orion architecture](images/orion-architecture.jpg)

从部署的角度看，自动化机器学习平台可以分为前台、中台、后台三部分，[部署图](https://drive.google.com/file/d/16HGDvJ90-MVmh1ElFh1EtQ8PMUUVQNYA/view)如下图所示。

![phoenix deplor](images/phoenix_deploy.jpeg)
### 分布式架构

分布式架构中有四个重要角色，分别是 `master`、`worker`、`tuning server` 和 `model manager`。用户通过 `client` 向平台输入训练数据，`client` 产生一个 `TrainJob` 输入给 `master`。其中 `master` 是一个后台 `grpc` 服务，主要用于对 `client` 提交的 `TrainJob` 进行调度。`worker` 是一系列训练任务的执行者，目前的 `worker` 有 `TrainWorker`、`HyperTrainWorker`、`EnsembleTrainWorker`、`PredictWorker` 四种，分别监听 `SingleQueue`、`HyperQueue`、`EnsembleQueue`、`PredictQueue` 四个队列，每个 `worker` 执行一个具体的训练任务。`ModelManager` 也是一个后台 `grpc` 服务，主要作用是将训练过程中的一些中间结果、监控信息及训练完成之后的模型的基本信息存储到数据库中，方便随时查看，也为后面的预测服务提供查询支持。

关于这四个角色，后面章节还会有详细介绍。

![master-worker-architecture](images/master-worker-architecture.jpg)

## 系统模块

### 数据预处理

在工程实践中，我们拿到的数据往往会存在有缺失值、异常值等，因此在应用机器学习算法之前需要进行数据预处理。自动化机器学习平台有一套标准的数据预处理流程，针对不同的任务和数据集属性分配不同的策略，包括多种数据集划分策略、多种缺失值处理策略、多种预处理方法等。

#### 数据集划分策略

* 固定比例切分

  固定比例切分是常用的一种切分方式，用户只需要指定好切分的训练集比例、验证集比例、测试集比例即可。在切分时可指定是否对数据进行 `shuffle` 处理，如果验证集和测试集都存在，还可指定是否对数据进行`re_fit` 训练，如果设置` re_fit` 为 `True`，平台会自动合并训练集和验证集，在合并后的数据上再次训练。

* K 折切分

  K 折切分 是交叉验证的一种方式，为了充分利用数据集对算法效果进行测试，将数据集随机切分为 K 个部分，每次将其中的 `K-1` 个部分作为训练集进行训练，而将剩下的那一部分作为测试集去验证模型效果。

* 留P法切分

  留P法切分 也是交叉验证的一种方式，假定数据集中有 N 个样本，设定一个 P 值，每次将 P 个样本作为测试样本，其他 `N-P`个样本作为训练样本。这样一共会得到 $ C_n^p $ 个 `train-test pairs`。

* 固定比例的时序切分

  固定比例时序切分是针对时间序列数据的切分方式，相对于非时间序列数据的切分，用户需要指定一个时间字段。然后平台会自动根据切分比例计算出训练集、验证集、测试集的起止时间，并将起止时间内的一段数据作为当前集合。

* 固定日期的时序切分

  固定日期时序切分也是针对时间序列数据的切分方式，相对于固定比例的时序切分方式，这种方式需要用户指定训练集、验证集和测试集的起止时间。

* 序列切分

  序列切分是针对序列数据的切分方式，这种方式需要用户指定训练集、验证集和测试集的起止索引。

* 指定索引切分

  指定索引切分方式是平台出于业务需求而设置的切分方式，主要用于集成训练中，指定一段集合中元素的索引列表，供 `sub model` 使用。

#### 缺失值处理策略

* 删除缺失值样本

  如果训练数据中含有缺失值，会对训练模型造成很大的干扰，因此在训练之前，有必要对缺失值数据进行处理。如果某条样本缺失值比例较大，那么可以选择直接删除该条样本。

* 填充缺失值

  对于含有少量缺失值的样本，可以选择某种策略对缺失值进行填充，平台提供了四种填充缺失值的方法。

  * 默认值填充

    这种方式下由用户指定一个默认的填充值，然后平台会自动地用这个默认值对含有缺失值的列进行填充。

  * 平均值填充

    这种方式下平台会自动对含有缺失值的列进行填充，填充值就是该列所有数值的平均值。

  * 中位数填充

    这种方式下平台会自动对含有缺失值的列进行填充，填充值就是该列所有数值的中位数。

  * 众数填充

    这种方式下平台会自动对含有缺失值的列进行填充，填充值就是该列所有数值的众数，所谓众数就是出现次数最频繁的那个数值。

#### 异常值处理策略

  异常值是数据集中存在的不合理的值，又称离群点，这些异常值将会影响模型的训练效果，带来噪声特征，因此有必要在训练前对这些异常值进行处理。平台目前的异常值处理，只是对特征值进行简单的统计分析，从而找出哪些值是不合理的。在找到这些异常值后，平台提供了两种方法进行处理：`TYPE_DROP_ROW` 和 `TYPE_COL_EXTREME` 。

* TYPE_DROP_ROW

  先计算出特征数据的平均值( `mean` )和方差( `std` )，根据提前设定的控制参数 `std_alpha` 计算出合理特征值的上下界，$$lower = mean - std * std\_alpha$$，$$upper = mean + std * std\_alpha$$，对于特征值小于 `lower` 或者特征值大于 `upper` 的行进行删除。

* TYPE_COL_EXTREME

  同样是先计算出特征数据的平均值( `mean` )和方差( `std` )，根据提前设定的控制参数 `std_alpha` 计算出合理特征值的上下界。但是和 `TYPE_DROP_ROW` 不同的是，`TYPE_COL_EXTREME` 对于小于 `lower` 的特征值都用 `lower` 的值去替换，对于大于 `upper` 的特征值都用 `upper` 的值去替换。

#### 预处理类型

* 特征二值化

  特征二值化的方法是根据一个阈值将特征的取值转化为 0 或 1，如果该特征的值大于阈值，就转化为 1，小于等于阈值的转化为 0。默认的阈值是 0，仅将所有的正数值转化为 1。在平台中用 `TYPE_BINARIZE_NUMERIC`表示。使用时传入两个参数，float 型的 `threshold` 表示转化的阈值大小，默认值是 0.0；还有一个布尔型的 `copy`，默认值是 True，表示对输入数据进行拷贝。如果设置为 False, 则会就地执行二值化的处理, 而不是先拷贝 input，前提是输入数据已经是 `numpy.ndarray` 。平台会自动记录二值化的预处理信息，即 `threshold` 和 `copy`的值，以便在 transform 的时候直接使用。

* 标准化

  标准化是将原始数据归一化成均值为 0、方差为 1 的数据，归一化公式为 $$X_{norm}=\frac{X - \mu}{\sigma}$$ ，其中 $\mu$ 和 $\sigma$ 分别是原始数据的均值和方差。在平台中用 `TYPE_STANDARDIZE_STANDARD`表示。使用时可以传入 `copy`, `with_mean`, `with_std` 三个参数，`copy` 是布尔型，默认值是 True，表示对输入数据进行拷贝。如果设置为 False，则会就地执行标准化的处理, 而不是先拷贝 input，前提是输入数据已经是 `numpy.ndarray`；`with_mean`也是布尔型，默认是 True，表示在标准化之前会对数据做中心化处理；`with_std` 也是布尔型，默认值是 True，表示会把数据 `scaler` 到标准差为 1 的范围。预处理完成后，平台会自动记录预处理信息，包括 `mean_`, `n_samples_seen_`, `var_`, `scale_`, `with_std`, `with_mean`, `copy`，以便在 transform 的时候直接使用。

* 区间缩放

  区间缩放是将特征值缩放到一个给定的范围内，使用区间缩放的目的包括实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。平台提供了两种区间缩放方法： `TYPE_STANDARDIZE_MIN_MAX` 和  `TYPE_STANDARDIZE_MAX_ABS`。

  * TYPE_STANDARDIZE_MIN_MAX

    `TYPE_STANDARDIZE_MIN_MAX` 将特征值缩放到一个指定的最小值和最大值之间，通常是是 [0, 1] 。缩放公式为$$X_{std}=\frac{X - X_{min}}{X_{max} - X_{min}}$$  ,  $$X_{scaled}=X_{std} * (max -  min) + min$$ , 其中 min 和 max 是要缩放到的区间范围，通常是 [0, 1]。在使用时可以传入两个参数：`feature_range` 和 `copy`，`feature_range`是一个元祖，默认是 (0, 1)，表示要缩放的范围； `copy` 参数含义同上。预处理完成后，平台会自动记录预处理信息，包括 `data_max_`, `scale_`, `var_`, `data_range_`, `n_samples_seen_`, `min_`, `data_min_`，`feature_range`, `copy`，以便在 transform 的时候直接使用。

  * TYPE_STANDARDIZE_MAX_ABS

    `TYPE_STANDARDIZE_MAX_ABS` 和 上面的 `TYPE_STANDARDIZE_MIN_MAX` 类似，但是它只通过除以每个特征的绝对值的最大值将训练数据特征缩放至  `[-1, 1]`  范围内。这意味着，训练数据应该是已经零中心化或者是稀疏数据，且不会破坏数据的稀疏性。在使用时可以传入参数`copy`，`copy` 参数含义同上。预处理完成后，平台会自动记录预处理信息，包括 `scale_`, `max_abs_`, `n_samples_seen_`, 及`copy`，以便在 transform 的时候直接使用。

* 特征编码

  很多时候，在我们拿到的数据集里，特征不都是连续的值，而是由某些离散化取值的数据组成。这些离散特征是无法直接被模型识别的，因此必须对这些离散特征进行数值化的编码。平台提供了多种针对离散化特征的编码方式，包括 `TYPE_INDEXER` ， `TYPE_ONE_HOT`， `TYPE_INDEXER_ONE_HOT`， `TYPE_CATEGORY_RATIO`。此外还提供了两种对日期特征的编码方式：`TYPE_DATETIME_TUPLE_ENCODE`,  `TYPE_DATETIME_ENCODE`。

  * TYPE\_INDEXER：对分类的 label 进行类别编码，使用时可以传入 `classes` 参数，类型是 `numpy.ndarray`，表示类别数组。预处理完成后，平台会自动记录预处理信息，即 `classes_` ，以便在 transform 的时候直接使用。
  * TYPE\_ONE\_HOT：对离散特征进行独热编码，使用时可以传入六个参数，分别是 `classes`，`n_values`，`categorical_features`，`dtype`，`sparse`，`handle_unknown`。预处理完成后，平台会自动记录预处理信息，包括 `classes_`, `feature_indices_`, `active_features_` , `n_values`，`n_values_`，  `categorical_features` , `dtype` ，`sparse`，及`handle_unknown`，以便在 transform 的时候直接使用。
    * `classes`的类型是 `numpy.ndarray`，表示类别数组。
    * `n_values` 表示每个特征的取值个数，可以为 `auto`、整数或者整数数组。`auto` 表示从训练集中获取特征值范围；若值为一个整数，则特征的取值必须在 `range(n_values)` 范围内；若值为整数数组，则特征 X[: i] 对应的分类特征有 `n_values[i]` 个取值，特征值的范围在 `range(n_values[i]` 内。
    * `categorical_features`：可能取值为 `all`, `indices` 数组或 `mask` 数组。若为 `all` 时，所有的特征都被视为分类特征。若为 `indices` ，则 `indices` 数组对应的特征被视为分类特征。若为 `mask` ，则表示特征长度数组，数组中元素为 `True` 或 `False`。
    * `dtype`：数值特征的类型，默认是 `numpy.float`。
    * `sparse`：布尔型，为 True 时，返回稀疏矩阵，否则返回数组。
    * `handle_unknown`：字符串类型，可以取值 `error` 或者 `ignore`，表示在 transform 时候如果遇到未知的分类特征，是抛出错误还是直接忽略。
  * TYPE_INDEXER_ONE_HOT: 和 TYPE_ONE_HOT 是一样的，只是基于业务进行区分。
  * TYPE_CATEGORY_RATIO：统计每个类别的样本所占的比例。可以传入参数 `counter_dict` ，表示每个类别出现的次数。预处理完成后，平台会自己记录预处理信息，包括 `counter_dict`  和 `total` ，以便在 transform 的时候直接使用。
  * TYPE_DATETIME_ENCODE: 对表示日期的字符串进行编码，需要传入日期格式参数 `time_format_str`。预处理完成后，平台会自动记录预处理信息，即 `time_format_str` 。

* 平滑处理

  平滑处理是针对时间序列数据的方法。大部分时间序列都存在一个重要问题：存在噪音，也就是某个值的大小随机变化。消除噪音或至少减小它的影响对时序分析很重要。平台支持三种平滑处理方法，差分平滑、移动平均平滑、指数平滑，分别对应着 `TYPE_DIFF`, `TYPE_ROLLING_MEAN`, `TYPE_EXPONENT_MEAN`。

  * `TYPE_DIFF`：对时间序列数据做差分平滑处理，在使用时需要传入三个参数： `time_field_name` 表示待处理的时间字段； `diff_order` 是一个 int 型整数，表示差分的阶数； `fill_value` 表示出现 NAN 之后的填充值。预处理完成后，平台会自动记录预处理信息，包括 `time_field_name`， `diff_order`， `fill_value`， `last_time`， `last_value`， 以便在 transform 的时候直接使用。

  * `TYPE_ROLLING_MEAN`：对时间序列数据做移动平均处理，在某个点的取值用过去一个窗口内的平均值来代替。在使用时需要传入三个参数： `time_field_name` 表示待处理的时间字段； `window` 是一个 int 型整数，表示计算统计量的观测值的数量即向前几个数据； `fill_value` 表示出现 NAN 之后的填充值。预处理完成后，平台会自动记录预处理信息，包括 `time_field_name`， `window`， `fill_value`，以便在 transform 的时候直接使用。

  * `TYPE_EXPONENT_MEAN`：对时间序列数据做指数加权平均，在使用时需要传入三个参数： `time_field_name` 表示待处理的时间字段； `com` 是一个 float 型浮点数，表示质心的衰减系数，可以用来计算平滑因子 `α` 的值， $$α = 1 / (1 + com)$$； `fill_value` 表示出现 NAN 之后的填充值。预处理完成后，平台会自动记录预处理信息，包括 `time_field_name`， `com`， `fill_value`，以便在 transform 的时候直接使用。


* 数学处理

  数学处理是对特征做一些简单的数学变换，如对数变，平方变换、平方根变换，分别对应着平台中的 `TYPE_LOG`, `TYPE_SQUARE`, `TYPE_SQRT`。

### 后台服务

自动化机器学习平台的运转依赖于后台的三个基础服务，分别是主服务( `ML Service`)，模型管理服务( `ModelManager Service`)，超参优化服务(  `Tuning Service`)。这三个基础服务进程常驻于系统后台，为平台的运转提供调度支持。每个 `service` 底层都是基于 `grpc + proto` 实现的，`proto` 文件中定义了每个 `service` 需要用到的服务接口，具体的方法需要在子类中进行实现。子类除了实现服务接口外，还会通过 `grpc.server` 类把 `service` 的接口注册到网络服务中，并通过网络地址暴露出去。

#### 主服务

主服务就是分布式架构中的 `Master` 的角色，主要对训练以及预测的 `Job` 进行调度。其类图结构如下所示。

![ml service class](images/ml_service.jpg)

#### 模型管理服务

模型管理服务就是分布式架构中的 `ModelManager` 角色，主要用于对训练过程产出的模型进行管理，主要包括模型训练的状态、模型组成、模型结果度量、模型存储路径等信息。同时，模型管理还为预测服务提供模型基本信息查询的支持。其类图结构如下图所示。

![model manager class](images/model_manager_service.jpg)

#### 超参数优化服务

超参数优化服务是进行自动化超参数训练的必备服务，可利用多种超参数优化算法对训练任务的参数空间进行搜索，以使指定的每个模型达到最好的效果。

其类图结构如下所示。

![tunign service class](images/tuning_service.jpg)

### 模型训练

#### 支持的模型

自动化机器学习平台目前集成了 Scikit-Learn 、StatsModels和Tenshorflow 三个主流框架，可用于训练多种机器学习模型，包括常见的统计学习模型和深度学习模型的训练，覆盖多种机器学习场景，包括分类、回归、聚类和时间序列分析。目前支持的模型一共包括以下 7 类 28 种算法：

<table>
   <tr>
      <td>算法类别</td>
      <td>算法名称</td>
      <td>问题分类</td>
   </tr>
   <tr>
      <td rowspan="2">逻辑回归算法</td>
      <td>传统逻辑回归( LR )</td>
      <td rowspan="2">分类问题</td>
   </tr>
   <tr>
      <td>梯度下降逻辑回归( SGD-LR )</td>
   </tr>
   <tr>
      <td rowspan="6">支持向量机算法</td>
      <td>核函数支持向量机分类( Kernel-SVC )</td>
      <td rowspan="3">分类问题</td>
   </tr>
   <tr>
      <td>线性支持向量机分类( Linear-SVC )</td>
   </tr>
   <tr>
      <td>梯度下降支持向量机分类( SGD-SVC )</td>
   </tr>
   <tr>
      <td>核函数支持向量机回归( kernel-SVR )</td>
      <td rowspan="3">回归问题</td>
   </tr>
   <tr>
      <td>线性支持向量机回归( Linear-SVR )</td>
   </tr>
   <tr>
      <td>梯度下降支持向量机回归( SGD-SVR  )</td>
   </tr>
   <tr>
      <td rowspan="3">贝叶斯算法</td>
      <td>高斯贝叶斯( Gaussian-NB )</td>
      <td rowspan="3">分类问题</td>
   </tr>
   <tr>
      <td>多项式贝叶斯( Multinomial-NB )</td>
   </tr>
   <tr>
      <td>伯努利贝叶斯( Bernoulli-NB )</td>
   </tr>
   <tr>
      <td rowspan="6">树型算法</td>
      <td>决策树分类( CART-DTC )</td>
      <td rowspan="3">分类问题</td>
   </tr>
   <tr>
      <td>XGBoost分类( XGB-DTC )</td>
   </tr>
   <tr>
      <td>随机森林分类( CART-RFC )</td>
   </tr>
   <tr>
      <td>决策树回归( CART-DTR )</td>
      <td rowspan="3">回归问题</td>
   </tr>
   <tr>
      <td>随机森林回归( CART-RFR )</td>
   </tr>
   <tr>
      <td>XGBoost回归( XGB-DTR )</td>
   </tr>
   <tr>
      <td rowspan="5">网络算法</td>
      <td>多层神经网络( DNN )</td>
      <td rowspan="5">分类问题</td>
   </tr>
   <tr>
      <td>卷积神经网络( CNN )</td>
   </tr>
   <tr>
      <td>递归神经网络( RNN )</td>
   </tr>
      <tr>
      <td>递归神经网络( LSTM )</td>
   </tr>
   <tr>
      <td>时间卷积网络( TCN )</td>
   </tr>
   <tr>
      <td rowspan="3">聚类算法</td>
      <td>K均值( KMeans )</td>
      <td rowspan="3">聚类问题</td>
   </tr>
   <tr>
      <td>小批量K均值( Mini Batch KMeans )</td>
   </tr>
   <tr>
      <td>DBSCAN</td>
   <tr>
      <td rowspan="3">时间序列算法</td>
      <td>移动平均模型( MA )</td>
      <td rowspan="3">时序问题</td>
   </tr>
   <tr>
      <td>自回归移动平均模型( ARMA )</td>
   </tr>
   <tr>
      <td>差分自回归移动平均模型( ARIMA )</td>
   </tr>
   </tr>
</table>


#### 模型训练方式

##### 普通训练

* 简介

  普通训练是指确定模型以及模型参数的训练，适用于有一定机器学习背景的人员，可以针对特定的数据自己指定合适的模型及模型参数。

* 类图设计

  普通训练的类图设计如下所示：

  ![single train class](images/single_train.jpg)

* 流程图

  普通训练的流程图如下所示，`TrainWorker` 负责监听 `SingleQueue` ，从中取得一个 `TrainJob` 然后开始训练，将训练过程中的一些评价指标及 `Log` 信息存储到 `HDFS` 中，训练结果会推送到一个 `ResultQueue` 中，`ModelManaager` 把训练完的模型信息保存到 `MongoDB` 数据库中。

  ![single train flow](images/single_train_flow.jpg)

##### 自动化超参训练

* 简介

  自动化机器学习平台除了支持上个章节所述的确定参数模型的训练以外，还支持对模型的参数进行优化搜索，即自动化超参数调优，避免了用户手动调参的低效。这种模式下，用户只需要上传数据，选择几个机器学习模型即可，然后平台会自动搜索出每个模型的最优参数。

* 自动化超参数优化算法

  平台目前支持5种自动化超参算法：网格搜索算法( `TYPE_GRID_SEARCH` )，随机搜索算法( `TYPE_RANDOM_SEARCH` )，基于序列模型的算法配置( `TYPE_SMAC` )，TPE算法( `TYPE_TPE` )，模拟退火算法( `TYPE_ANNEAL` )，涵盖了基础搜索方法与基于采样的方法。

  * `TYPE_GRID_SEARCH` ：网格搜索，基础搜索方法的一种，将所有候选的参数的可能的取值形成一个多维的网格，循环遍历每个网桥，即尝试每一种可能的参数组合，取表现最好的参数组合。

  * `TYPE_RANDOM_SEARCH`：随机搜索，基础搜索方法的一种。网格搜索只能在给定的参数列表中进行组合选择，穷举复杂度高，且一般参数的候选值都是连续的，且参数较多时，容易出现只在局部搜索的情况，因此实际中使用较少。随机搜索( `Random Search` )是加拿大蒙特利尔大学的两位学者 Bergstra 和 Bengio 在他们 2012 年发表的[论文](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)中提出的，表明随机搜索比网格搜索更高效，见下图。

    ![grid_random_compare](images/grid_random.png)

  * `TYPE_SMAC` :  `SMAC`  全称是 `Sequential Model-based Algorithm Configuration` ，属于基于模型的零阶优化，基于采样的方法的一种。它是用于在一组实例中优化任意算法参数的算法配置工具，包括但不限于硬组合问题求解器的参数优化和各种机器学习算法的超参数优化。 核心包括贝叶斯优化、随机森林模型的和实例上的简单竞赛机制，有效地决定了两个配置的效果。

  * `TYPE_TPE` : `TPE` 全称是 `Tree-structured Parzen Estimator Approach` ，属于基于模型的零阶优化，基于采样的方法的一种。在超参搜索时，`TPE` 算法从过去的搜索结果中利用 `Kernel density estimation` 构建出参数的树结构的概率密度模型，通过在概率密度模型中随机 sampling ，并通过最大化预期提升（EI）来决定下一组待评估的超参数，相较于高斯过程，TPE 算法是相当高效的，因为基于高斯过程的方法的复杂度是观测历史数据量与参数数目（维度）的三次方，而TPE 则是线性的。

  * `TYPE_ANNEAL`:  模拟退火同样是使用采样的方法，属于局部搜索的方法。随机选择一个解作为开始值，下面每次迭代过程中，在已观察解的领域内按照其原始分布sample一个值做为下一次迭代的值，每次迭代过程中邻域的范围是递减的，过程就像是退火的过程一样，最后会慢慢集中在最优解的附近搜索，直到迭代搜索结束，本质上是优化后的随机搜索。

* 类图设计

  自动化超参训练的类图设计如下所示：

  ![hyper train class](images/hyper_train.jpg)

* 流程图

  自动化超参训练的流程图如下所示，`HyperTrainWorker` 负责监听 `HyperQueue` ，从中取得一个 `HyperTrainJob` 然后开始超参训练，训练过程中，通过与`TuningServer`通信可以根据指定的超参算法，为 `HyperTrainJob` 的每个模型产生一个具体参数的 `SubTrainJob`， 并推送到 `SingleQueue` 中。 `TrainWorker` 负责训练每个 `SubTrainJob` ，训练完成后，将 `SubTrainResult`  推送到 `ResultQueue` 中，同时将训练完成的 `SubTrainJob`  回推到 `HyperQueue` 中。每一次 `SubTrainJob`  的训练代表一次超参搜索过程。`HyperTrainWorker`接收到训练完成的 `SubTrainJob`，继续通过`TuningServer`来获得下一次搜索的参数，直到达到最大搜索次数后结束超参训练。

   超参优化架构图如下：

  ![hyper train flow](images/hyper_train_flow.jpg)

  HyperTrainWorker 超参优化时的执行流程图如下：

  ![hyper train worker flow](images/hyper_train_worker.png)

##### 集成训练

* 简介

  集成方法是将几种机器学习算法组合成一个预测模型的训练方式，以达到减小方差、偏差或改进预测的效果。本平台的集成训练是基于 `stacking` 的集成学习技术，通过元分类器或元回归( `Meta-Model` )聚合多个基分类或回归模型( `Base-Model` )。其中基模型( `Base-Model` ) 基于完整的训练数据进行训练，然后元模型( `Meta-Model` )基于基模型的输出结果进行训练。

  下图展示了平台中基于 `Base-Meta` 结构的集成训练的流程，其中 `Bi` 代表一个基模型，基模型分别对验证集、测试集做出预测，然后汇总每个基模型在验证集上的测试结果，作为新的特征数据，输入给元模型 `Meta Model` 继续训练，`Meta Model` 的预测结果将作为最终的结果。

  ![stacking ensemble](images/stacking_ensemble.jpg)

* 类图设计

  集成训练的类图设计如下图所示：

  ![ensemble train class](images/ensemble_train.jpg)

* 流程图

  集成训练的流程图如下所示，`EnsembleTrainWorker` 负责监听 `EnsembleQueue` ，从中取得一个 `EnsembleTrainJob` ，根据指定的 `Base Model`  及 `Meta Model` 信息，将 `EnsembleTrainJob`  拆分成多个基模型 `SubBaseJob`， 并推送到 `SingleQueue` 中。 `TrainWorker` 负责训练每个 `SubBaseJob` ，训练完成后，将 `SubTrainResult`  推送到 `ResultQueue` 中。`EnsembleTrainWorker` 会汇总每个 `SubBaseModel` 的预测结果，生成一份新的训练数据。 然后将指定的的 `MetaJob`  推送到  `SingleQueue` 中,  `TrainWorker` 取得  `MetaJob`  ，然后在生成的新数据上再次训练。

  ![ensemble train flow](images/ensemble_train_flow.jpg)


### 模型评估

模型评估模块支持多种机器学习场景的评估方式，包括分类、回归、聚类和实际序列。

#### 分类评估

* 准确率

  准确率是预测正确的正例数据占预测为正例数据的比例，计算公式是 $ P=TP/(TP+FP) $。平台针对分类问题，提供了细节全面的准确率衡量指标。包括衡量总体准确率的 `TYPE_ACC_MEAN` 和针对每一类的预测准确率 `TYPE_PRECISION_EACH`。

* 召回率

  召回率是预测为正例的数据占实际为正例数据的比例，它与准确率二者相辅相成，计算公式是$ R=TP/(TP+FN) $。平台同样也提供了针对每一类的预测召回率 `TYPE_RECALL_EACH`。

* F1 分数

  F1值 (F1-score) 是用来平衡准确率和召回率，计算方式是准确率和召回率的调和均值，公式是 $$F1=\frac{2}{\frac{1}{P} + \frac{1}{R}}=\frac{2 * P * R}{P+R}  $$。平台同样也提供了针对每一类的 F1 分数 `TYPE_F1_EACH`。

* ROC 曲线

  ROC 曲线全称是`接收者操作特征曲线（receiver operating characteristic curve）`，是反映敏感性和特异性连续变量的综合指标，roc 曲线上每个点反映着对同一信号刺激的感受性。在平台中用 `TYPE_ROC` 表示，主要针对于二分类问题。

* AUC

  AUC 全称是 `(Area Under Curve)`，意思是 `ROC` 曲线下方的面积。很多时候 `ROC` 曲线并不能很好的说明哪个分类器的效果更好，而 `AUC` 作为一个具体的数值，更加清晰明了，一般 `AUC` 值更大的分类器效果更好。此外 `AUC` 值在平台中也可以作为训练时的 `loss` 目标，来优化训练过程。

* KS

  KS 全称是 `(Kolmogorov-Smirnov)`，主要对模型风险区分能力进行评估，衡量的是好坏样本累计分布之间的差值。 好坏样本累计差异越大，KS 指标越大，那么模型的风险区分能力越强。

#### 回归评估

* MAE

  MAE (Mean Absolute Error) 叫做平均绝对误差，是绝对误差的平均值，能更好地反应预测值误差的实际情况。

* MSE

  MSE (Mean Squared Error) 叫做均方误差，是回归问题中常用的一个指标，用来衡量观测值同真实值之间的偏差。在平台中也可以作为回归问题的 `loss` 目标，来优化训练过程。

* RMSE

  RMSE (Root Mean Squard Error) 叫做均方根误差，其实就是将 `MSE` 值开根号，实质是一样的，同样也用来衡量观测值同真实值之间的偏差。

* MSLE

  MSLE (Mean Squared Log Error) 在计算时将真实值 `y_true` 与预测值 `y_pred` 做了 `log` 的处理，然后就和 `MSE` 的计算过程是一样的。这样当数据中有少量的值和真实值差值较大的时候，使用 `log` 函数能够减少这些值对于整体误差的影响。

* MDAE

  MDAE (Median Absolute deviation) 叫做绝对中位差，计算方式是原数据减去中位数后得到的新数据的绝对值的中位数，MDAE常用来估计标准差。

* R Squared

  R Squared 方法是将预测值跟只使用均值的情况下相比，看能好多少。其区间通常在 `[0,1]` 之间。`0` 表示还不如什么都不预测，直接取均值的情况，而 `1` 表示所有预测跟真实结果完美匹配的情况。

* MAPE

  MAPE (mean absolute percentage error) 叫做平均绝对百分比误差。 `MAPE` 不仅考虑预测值与真实值的误差，还考虑了误差与真实值之间的比例。

#### 聚类评估

* ARI

  ARI (Adjusted Rand Index) 指标需要事先知道样本的真实标签，是用来衡量聚类得到的标签分布和真实标签分布相似性的函数， ARI 值的范围是 `[-1, 1]`，负的结果都是较差的，说明标签是独立分布的。相似分布的ARI  值结果是正的，`1` 是最完美的结果，说明两种标签的分布完全一致。

* FMI

  FMI (Fowlkes-Mallows Scores) 指标同样也需要事先知道样本的真实标签，用来衡量两个聚簇之间的相似性。FMI 值的范围是 `[0,1]`，数值越高表明两个簇之间越相似。

* SC

  SC (Silhouette Coefficient)，叫做轮廓系数，它结合了内聚度和外聚度两种因素。可以用来在相同原始数据的基础上用来评价不同算法、或者算法不同运行方式对聚类结果所产生的影响。SC 值的范围是 `[-1,1]`，越趋近于 `1` 代表内聚度和分离度都相对较优。

* CHI

  CHI (Calinski-Harabaz Index) 指标不需要知道样本的真实标签，通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，CHI 指标由分离度与紧密度的比值得到。因此，CHI 越大代表着类自身越紧密，类与类之间越分散，代表更优的聚类结果。

#### 时间序列评估

上述的回归评价指标均可以用于时间序列预测类问题的评估，此外平台还专门提供了几个针对时间序列问题的评估指标。

* 残差平方和

  残差平方和是预测值与真实值之间的误差平方和。

* 残差图

  残差的可视化展示。

###  预测服务

* 简介

  用户通过自动化机器学习平台训练完成模型后，可以通过平台的预测功能对新数据进行批量预测。预测模块负责加载用户指定的模型并执行提交的预测任务，该模块具备以下四个特点：任务提交和执行的异步化、多模型共存、模型单例性、多模式预测请求服务。

* 类图设计

  ![predict class](images/predict.jpg)

* 流程图

  预测的主要流程如下所示：

  1. 用户同过 `model_id` 申请加载某一个已经训练好的模型实例，系统将这个加载请求放在等待队列中并为该模型分配一个预测任务等待队列，同时系统返回获取预测结果的路径。
  2. 用户提交一个或多个训练任务到系统分配的预测任务等待队列中。
  3. 用户提交一个模型撤销请求到预测任务等待队列，平台会自动判断是否删除该 `model_id` 对应的模型实例。
  4. 最后用户从系统返回的结果路径中获取预测结果。

  总结预测模块的流程图如下所示：

  ![predict flow](images/predict_flow.jpg)

### 分布式日志收集检索

#### 背景

平台每天会产生大量的日志（一般为流式数据，如，模型的训练，模型预测，online serving等），处理这些日志需要特定的日志系统，一般而言，这些系统需要具有以下特征：

- 帮助开发人员定位错误，一站式查看在各个机器上任务日志。
- 构建应用系统和分析系统的桥梁，并将它们之间的关联解耦。
- 支持近实时的在线分析系统和类似于Hadoop之类的离线分析系统。
- 具有高可扩展性。即：当数据量增加时，可以通过增加节点进行水平扩展。

#### 技术栈

通过[对比](/2018/11/30/%E5%9F%BA%E4%BA%8EELK%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E6%97%A5%E5%BF%97%E6%94%B6%E9%9B%86%E6%A3%80%E7%B4%A2%E7%B3%BB%E7%BB%9F/#%E6%97%A5%E5%BF%97%E6%94%B6%E9%9B%86%E6%96%B9%E6%A1%88%E9%80%89%E6%8B%A9)已有的日志收集方面的技术方案，最后我们选择了 ELK (**E**lasticsearch + **L**ogstash + **K**ibana) 作为最终的技术方案。ELK 已经成为目前最流行的集中式日志解决方案，它主要是由`Beats`、`Logstash`、`Elasticsearch`、`Kibana`等组件组成，来共同完成实时日志的收集，存储，展示等一站式的解决方案。

1. Filebeat：Filebeat相当于标准日志系统中的 agent，是一款轻量级，占用服务资源非常少的数据收集引擎，它是ELK家族的新成员，可以代替 `logstash` 作为在应用服务器端的日志收集引擎，支持将收集到的数据输出到Kafka，Redis等队列。
2. Logstash：Logstash 相当于标准日志系统中的 collector，数据收集引擎，相较于Filebeat比较重量级，但它集成了大量的插件，支持丰富的数据源收集，对收集的数据可以过滤，分析，格式化日志格式。
3. Elasticsearch：Elasticsearch 相当于标准日志系统统中的 store，分布式数据搜索引擎，基于Apache Lucene实现，可集群，提供数据的集中式存储，分析，以及强大的数据搜索和聚合功能。
4. Kibana：数据的可视化平台，通过该web平台可以实时的查看 Elasticsearch 中的相关数据，并提供了丰富的图表统计功能。

ELK的安装配置请转[这篇博客](/2018/11/30/%E5%9F%BA%E4%BA%8EELK%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E6%97%A5%E5%BF%97%E6%94%B6%E9%9B%86%E6%A3%80%E7%B4%A2%E7%B3%BB%E7%BB%9F/#ELK%E7%9A%84%E5%AE%89%E8%A3%85%E3%80%81%E9%85%8D%E7%BD%AE%E4%B8%8E%E4%BD%BF%E7%94%A8)。

#### 日志收集架构

我们用`filebeat` 取代 `logstash` 做为日志收集器，使用 `logstash`分析、处理日志数据，使用ES检索日志，通过 nginx 反向代理访问`kibana`对日志进行查询。另外，在 k8s 部署时，单独创建一个日志收集组件跟app的docker容器一起运行在同一个pod中，整个架构如下：

![log architecture](images/log_architecture.png)

#### 监控

ES Monitor用于监控 `elasticsearch`，有两个作用：

1. 通过监控 `elasticsearch` 中日志的情况，对平台中的训练任务近实时邮件报警，并在邮件中提供相关错误链接，以方便错误定位与调试。

   ![es_monitor_alert_email](images/es_monitor_alert_email.png)

   邮件中的错误是按照时间戳 Sorted Descending。

2. 对 ES 中的索引进行维护，删除过期的日志索引数据 (目前自动删除半年前的日志数据)。

另外，`kibana`本身也可以可视化监控 ELK Stack 的运行情况，只需在 `filebeat`, `logstash`, `elasticsearch` 中打开 `xpack` 中的监控功能，可以监控到的如如内存占用，事件处理的频率，出错的情况，机器负载等，可参考下图。

![kibana_monitor_cluster](images/kibana_monitor_cluster.png)

![kibana_monitor_overview](images/kibana_monitor_overview.png)

#### 日志可视化

我们除了可以在 `kibana`使用`lucene`的语法对日志进行查询外，还可以通过可视化对日志的情况有一个直观的了解。

下图是查看 id 包含`3ff17b42-f842-11e8-955a-f8bc12a1071b`的任务各阶段所用时间的折线图。

![kibana_visualize_time_spend_per_task](images/kibana_visualize_time_spend_per_task.png)

下图显示一段时间内平台发生的错误的分布饼图。

![err pie](images/err_pie.png)

用户可根据[教程](https://www.elastic.co/guide/en/kibana/current/visualize.html)进一步修改或者添加自己的可视化元素与仪表盘。

### 用户交互

#### 前端设计

自动化机器学习平台提供了简单易用的用户交互界面，是离用户最近的、用户最直观看到的界面，专注于数据展示和交互体验，实现数据上传、提交训练参数、结果可视化展示等功能。

自动化机器学习平台前端以 Web端 的形式提供给用户使用，具备较为完善的技术体系，包含技术工具栈（技术选型和基础工具设施），构建工具栈，测试工具栈。使用 `TypeScript` 进行开发，包含更丰富的工程化特性，开发体验得到显著提升，静态检查的能力也使得代码更加安全、可靠。前端技术栈如下图所示：

![web develop stack](images/web-dev.png)

开发工具栈中，为了更有效的创建用户交互界面，带来更好、更流畅的用户体验，采用当下主流的 `React` 库。它由 Facebook、Instagram 和一个由个人开发者和企业组成的社群维护，是优秀的 JavaScript 开源库。由于 `Orion前端` 服务于不同的用户身份，存在较多的于服务器交互，同时为了满足持续的业务需求，我们选用 MobX 作为状态管理工具，它通过透明的函数响应式编程使得状态管理变得简单和可扩展。

数据通讯方面使用 `Axios` 库，其支持 `Promise API` 的特性，让数据响应处理更加便捷和可靠。同时，为了及时地向用户反馈训练任务的最新进程，使用 WebSocket 技术实现前端和中台的双向通讯。

为了提供企业级中后台产品的交互语言和视觉风格，选用蚂蚁金服技术体验设计 `Ant-Design` 样式库 ，其拥有开箱即用的高质量 React 组件，可以让我们更高效地完成前端页面的搭建，更专注于自动化学习平台业务的开发。

构建工具栈中，通过 `Webpack` 进行项目的构建和打包，`Yarn` 作为团队统一的包管理工具，`Sass` 作为 CSS 的预编译工具提升 CSS 代码的可维护性，`Babel` 作为 ES6 的编译工具，这样我们代码里可以用到 ES6 的一些新特性和语法糖，`TSLint` 作为团队的代码检查工具保证代码的规范一致。同时借助一些开源的自动化测试工具提升开发阶段的代码质量。

基于此技术体系，前端工程的目录结构如下:

```bash
.
├── Dockerfile
├── build               # 构建脚本
├── config              # 配置，包含构建等配置
├── dist                # 构建后的
├── public              # 静态资源文件夹
│   ├── favicon.ico     # Favicon
│   └── index.html
├── src
│   ├── common          # 表单选型、ts 语法定义等
│   │   └── nav.ts      # 路由配置
│   ├── components      # 业务通用组件
│   ├── containers      # 页面内容
│   │   ├── App         # App 框架
│   │   ├── Auth        # 注册、登录系统
│   │   ├── Data        # 数据管理
│   │   ├── Main        # 页面布局
│   │   ├── Predict     # 预测任务管理
│   │   ├── Train       # 训练任务管理
│   │   └── User        # 用户管理
│   ├── locales         # 国际化资源
│   ├── stores          # 状态管理
│   ├── styles          # 页面内容样式
│   ├── utils           # 工具库
│   └── index.tsx       # 入口文件
├── tests               # 测试工具
├── typings             # ts 声明文件
├── README.md
├── package.json
└── yarn.lock
```

####  中台设计

`Orion 中台` 为用户进行自动化学习提供桥梁，注重于逻辑实现和数据存储，实现用户权限认证、模型保存、组织 schema 并提交训练、即时同步任务状态等能力。

在 Orion 产品中，前端和中台的代码放置在同一个 `Git` 仓库下，但同时由于前端和中台的关注点不同，采用前后端分离的开发模式。整体目录结构如下。

```bash
.
├── app                 # 业务
│   ├── auth            # 用户认证
│   ├── data            # 数据管理
│   ├── io              # IO 相关（HDFS、JSON）
│   ├── manage          # 用户管理
│   ├── predict         # 预测模块
│   ├── status          # 服务健康检测
│   ├── train           # 训练模块
│   └── utils           # 业务工具库
├── base                # 基础组件（配置、数据库）
├── common_proto        # Protobuf
├── docker              # 容器化部署脚本
├── fe                  # 前端代码，具体见目录结构
├── main                # 后端服务入口（包含 be、monitor、ws 服务）
├── migration           # 数据库迁移脚本
├── models              # 数据 Models
├── services
│   ├── monitor         # 任务进度监控
│   └── ws              # WebSocket 服务
├── tests               # 测试
├── utils               # 工具库
└── venv                # 虚拟环境
```

中台部分使用 `Python` 语言开发，由三个服务组成：`orion_be`、`orion_monitor` 和 `orion_ws` 。以下分别对这三个服务进行介绍。

* orion_be

`orion_be` 提供用户管理，训练任务管理，预测任务管理等接口服务，采用微框架 `Flask` 进行开发。框架本身不重，但很强大，给开发者足够的自由和选择，通过拓展，可以很轻松的实现权限校验、表单校验、上传处理、数据库整合等业务需要实现的功能。

数据存储方面使用的是开源的 `PostgreSQL` ，是一个自由的对象-关系数据库服务器（数据库管理系统）。为了统一方便安全的管理**业务 Model 对象**与**数据库**之间的关系，采用 `ORM（Object-Relational Mapping）` 框架 `SQLAlchemy`，把关系数据库的表结构映射到对象上。同时，随着项目业务需求的不断变更，数据库表结构的修改就难以避免，为了对数据库的修改加以记录和控制，便于项目版本管理和随意升级和降级，使用了 `Alembic` 进行数据库迁移和版本管理。

* orion_monitor

`orion_monitor` 服务基于 `Python` 任务调度框架 `APScheduler` 实现，定时发起查询任务最新进度的接口请求，通过过滤、处理接口返回的数据，更新最新的任务状态到数据库，方便前端直接查询。同时借助该框架，可以将任务存储到数据库中，实现任务的持久化。

在向后台模型管理服务发起接口请求的时候，可能因为硬件故障、程序bug、超出设计承载的高并发请求等问题，造成服务不可用。当发生这种情况时，由于 `orion_monitor` 服务定时发起查询任务，会产生大量等待线程消耗系统资源，同时不断的重试可能放大不可用状态，造成服务雪崩。为了避免造成更大范围的故障，程序应该主动降级。基于此，我们使用**熔断器模式**（`Circuit Breaker Pattern`）预防这种问题，提高应用的稳定性和灵活性。

我们使用状态机实现 `Circuit Breaker`，它有以下三种状态：

* 关闭（Closed）: 应用程序请求能够直接通过熔断器。如果在某时段内，失败率达到阈值，则会转到开启状态。
* 开启（Open）: 调用请求会立即失败并且抛出异常。同时，启用一个超时计时器，当计时器到时间的时候，会转到半开启状态，进行有限尝试。
* 半开启（Half-Opened）: 只允许限量调用这个操作，如果成功率大于阈值，`Circuit Breaker` 就会假定故障已经恢复，转换到关闭状态，并重置失败次数；如果成功率没有达到阈值，就认为故障仍然存在，就会转换到开启状态，并开启计时器。

  `Circuit Breaker` 状态图如下所示：
  ​			 ![Circuit Breaker](images/circuit_breaker.jpg)

* orion_ws

`orion_ws` 服务基于 `aiohttp` ，提供了多用户高并发的 `WebSocket` 服务支持。在用户将前端页面停留在任务列表的情况下，页面定时触发同步任务状态的 `WebSocket` 事件，中台接收到该事件之后，就会查询数据库，返回最新更新的任务状态。

可以发现 `orion_monitor` 从后台拉取最新的数据并持久化到数据库， `orion_ws` 服务和前端 `socket` 技术实现双向通讯，可以将训练任务和预测任务的最新进程及时地反馈给用户。

#### 前端展示

页面的整体布局统领整个应用的界面，是一个产品最外层的框架结构，包含侧边栏、页脚以及内容等。通常布局是和路由系统紧密结合，为了统一方便的管理路由和页面的关系，使用了中心化的方式，将配置信息统一抽离到 `src/common/nav` 下，并拓展了一些关于全局菜单的配置，如权限控制字段等；布局和菜单实现的代码放置在 `src/containers/Main` 下。

![web ui](images/web-ui.png)

新建页面的时候，需要考虑页面的具体业务，考虑放置在哪个路由下合适，页面代码在 `src/container` 的位置，几乎与路由的配置的层级路径保持一致，同时，也需要注意在 `src/containers/Main` 里做路由的映射，然后就可以到新建的 `NewPage.tsx` 写业务代码了。

对于一些可能被多出引用的功能模块，应该提炼成业务组件统一管理，这里我们放置到 `src/components` 文件夹下。这类组件只负责独立的一块，功能稳定，没有单独的路由配置，也有可能是纯组件，仅受父组件传递的参数控制。

Orion 前端技术体系还有一段路要走，未来还会继续优化技术体系，更好的服务于业务、支撑业务的快速发展。



