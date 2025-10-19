import gpytorch
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # 直接导入 tqdm 类
# 在现有代码中添加高斯过程多任务回归模型类
class MultitaskGPRegressor:
    def __init__(self, input_dim, output_dim, num_tasks=None, learning_rate=0.1, 
                 training_iterations=50, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks if num_tasks else output_dim
        self.learning_rate = learning_rate
        self.training_iterations = training_iterations
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.likelihood = None
        self.scaler = None
        self.loss_history = []

    def fit(self, X, y, verbose=True):
        # 数据预处理
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # 转换为pytorch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 初始化似然和模型
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)
        self.model = MultitaskGPModel(
            X_tensor, y_tensor, self.likelihood
        ).to(self.device)
        
        # 训练模式
        self.model.train()
        self.likelihood.train()
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 边际对数似然
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # 训练循环
        with tqdm(total=self.training_iterations, desc="GP训练进度", unit="iter") as pbar:
            for i in range(self.training_iterations):
                optimizer.zero_grad()
                output = self.model(X_tensor)
                loss = -mll(output, y_tensor)
                loss.backward()
                optimizer.step()
                
                self.loss_history.append(loss.item())
                if verbose and (i + 1) % 10 == 0:
                    print(f'Iter {i + 1}/{self.training_iterations} - Loss: {loss.item():.3f}')
                pbar.update(1)
        
        return self

    def predict(self, X):
        if self.model is None or self.likelihood is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # 数据预处理
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 评估模式
        self.model.eval()
        self.likelihood.eval()
        
        # 预测
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_tensor))
            return predictions.mean.cpu().numpy()  # 返回均值作为预测值

    def get_loss_history(self):
        return self.loss_history

# 高斯过程多任务模型定义
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks if num_tasks else train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=self.num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)