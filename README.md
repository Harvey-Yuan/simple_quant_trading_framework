## 目录

- [目录](#目录)
- [文件树介绍](#文件树介绍)
- [工作流](#工作流)
- [附录文件](#附录文件)
  - [AWS 部署](#aws-部署)

## 文件树介绍

| 文件夹名           | 描述                                                                              |
| ------------------ | --------------------------------------------------------------------------------- |
| `doc_resources`    | 所有用于展示和记录文档的储存地点。                                                |
| `wheels`           | 一些公共的 API 和 config 储存的地方，database 的 API 开发完成后会移动到 wheels 中 |
| `UI`               | 展示 UI 的地方                                                                    |
| `main`             | 是策略主程序运行的地方。                                                          |
| `main_prototypes`  | 预先编写好主程序的原型。简化主程序逻辑。                                          |
| `data_develop`     | 开发新数据获取和特征工程的文件                                                    |
| `model_develop`    | 开发机器学习/深度学习模型的地方                                                   |
| `strategy_develop` | 开发策略，回测策略的地方                                                          |

## 工作流

1.首先是开发流程，可以在 data_develop 中开发数据，在 model_develop 中开发模型，或者是在 strategy_develop 中开发策略。

2.其次是测试上线。main_prototypes 中集合了策略的原型，以其为基础，可以开发 main 文件夹中的策略。main 中的策略完成后，直接整个文件夹部署到 AWS，为每一个策略设置定时任务。关于其次是测试上线。main_prototypes 开发流程可以看[main_prototypes 说明](main_prototypes/prototypes.md),main 开发可以看[main 说明](main/main.md)

## 附录文件

### AWS 部署

如果对部署 AWS 任务有问题，可以看[AWS 使用指南](doc_resources/AWS_guide.md)
