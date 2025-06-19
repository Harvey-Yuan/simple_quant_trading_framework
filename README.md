## Table of Contents

- [Table of Contents](#table-of-contents)
- [File Tree Introduction](#file-tree-introduction)
- [Workflow](#workflow)
- [Appendix Files](#appendix-files)
  - [AWS Deployment](#aws-deployment)

## File Tree Introduction

| Folder Name        | Description                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `doc_resources`    | Storage location for all display and documentation files.                                                    |
| `wheels`           | Storage place for common APIs and config, database API will be moved to wheels after development is complete |
| `UI`               | Place to display UI                                                                                          |
| `main`             | Where the strategy main program runs.                                                                        |
| `main_prototypes`  | Pre-written prototypes of the main program. Simplifies main program logic.                                   |
| `data_develop`     | Files for developing new data acquisition and feature engineering                                            |
| `model_develop`    | Place to develop machine learning/deep learning models                                                       |
| `strategy_develop` | Place to develop strategies and backtest strategies                                                          |

## Workflow

1. First is the development process, you can develop data in data_develop, develop models in model_develop, or develop strategies in strategy_develop.

2. Next is testing and deployment. main_prototypes integrates strategy prototypes, and based on these, you can develop strategies in the main folder. After the strategies in main are completed, deploy the entire folder directly to AWS and set up scheduled tasks for each strategy. For the development process, you can see [main_prototypes documentation](main_prototypes/prototypes.md), and for main development, see [main documentation](main/main.md)

## Appendix Files

### AWS Deployment

If you have questions about deploying AWS tasks, you can see [AWS Usage Guide](doc_resources/AWS_guide.md)
