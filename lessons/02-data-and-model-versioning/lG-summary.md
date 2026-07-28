# Summary

## ML Projects Components

A typical ML project consists of several key components that all need version control:

1. **Code**: Model training, evaluation, prediction code
2. **Data**: Raw data, processed data, features
3. **Models**: Trained model artifacts
4. **Configurations**: Hyperparameters, settings
5. **Documentation**: READMEs, experiments logs
6. **Environment**: Dependencies, container images

## Next Steps

After understanding data versioning, the next topics to explore:

### Golang

Learn Go for building high-performance ML systems and tools.

### ML Project Structure

Organize your ML projects for maintainability and collaboration:
- Standardized directory structures
- Clear separation of concerns
- Reproducible environments
- Documentation standards

### Cookiecutter Data Science

Use project templates to ensure consistency across ML projects.

![Cookiecutter Data Science](/images/data-and-model-versioning/cookiecutter.png)

From https://cookiecutter-data-science.drivendata.org/

**Benefits:**
- Quick project setup
- Best practices built-in
- Consistent structure across projects
- Easy to share and reuse

## Productionising ML Models

Taking ML models from experimentation to production involves several stages:

1. **Experiment**: Try different approaches and algorithms
2. **Validate**: Ensure model performance meets requirements
3. **Package**: Create a deployable artifact
4. **Deploy**: Put the model into a production environment
5. **Monitor**: Track model performance in production
6. **Maintain**: Update and improve the model over time

![Google MLOps](/images/data-and-model-versioning/google-mlops.png)

From https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

### Key Considerations

- **Reproducibility**: Can you recreate the exact same model?
- **Scalability**: Can the model handle production workloads?
- **Monitoring**: How do you know if the model is performing well?
- **Versioning**: How do you manage different model versions?
- **Rollback**: Can you quickly revert to a previous version if needed?

## Course Recap

This lecture covered:

1. **Why data versioning matters** for ML projects
2. **Multiple levels of data versioning** from simple to specialized
3. **Data version operations** with DVC
4. **Storage considerations** and strategies
5. **Data stages** using medallion architecture
6. **ML-specific considerations** for data pipelines and splits

## Resources

- [DVC Documentation](https://dvc.org/doc/)
- [ML Engineering Book, Chapter 3.11](http://www.mlebook.com/wiki/doku.php)
- [ML-Ops.org: Data Engineering Pipelines](https://ml-ops.org/content/three-levels-of-ml-software)
- [LakeFS Blog](https://lakefs.io/blog/data-versioning/)
- [Git LFS](https://git-lfs.com/)
