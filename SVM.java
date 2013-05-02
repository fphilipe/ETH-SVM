package org.ethz.las;

import java.util.*;

public class SVM {

  // Hyperplane weights.
  RealVector weights;

  public SVM(RealVector weights) {
    this.weights = weights;
  }

  /**
   * Instantiates an SVM from a list of training instances, for a given
   * learning rate 'eta' and regularization parameter 'lambda'.
   */
  public SVM(List<TrainingInstance> trainingSet, double lambda, int T) {
    int dimension = trainingSet.get(0).getFeatureCount();
    int trainingSetSize = trainingSet.size();
    int K = trainingSetSize;

    double[] w = new double[dimension];
    for (int i = 0; i < dimension; i++)
      w[i] = 1.0 / Math.sqrt(lambda) / Math.sqrt((double)trainingSetSize);
    weights = new RealVector(w);

    for (int t = 1; t < T; t++) {
      RealVector gradientT = weights.scale(lambda);

      List<TrainingInstance> A = trainingSet;
      int lengthA = A.size();
      double eta = 1.0 / t / lambda;
      double sumFactor = eta / lengthA;

      for (int k = 0; k < K; k++) {
        RealVector x = A.get(k).getFeatures();
        int y = A.get(k).getLabel();

        if (x.dotProduct(weights) * y < 1)
          gradientT.add(x.scale(-y * sumFactor));
      }

      weights.add(gradientT.scale(-eta));
      double minFactor = 1.0 / Math.sqrt(lambda) / weights.getNorm();
      double weightsFactor = Math.min(1, minFactor);
      weights.scaleThis(weightsFactor);
    }
  }

  /**
   * Instantiates SVM from weights given as a string.
   */
  public SVM(String w) {
    List<Double> ll = new LinkedList<Double>();
    Scanner sc = new Scanner(w);
    while(sc.hasNext()) {
      double coef = sc.nextDouble();
      ll.add(coef);
    }

    double[] weights = new double[ll.size()];
    int cnt = 0;
    for (Double coef : ll)
      weights[cnt++] = coef;

    this.weights = new RealVector(weights);
  }

  /**
   * Instantiates the SVM model as the average model of the input SVMs.
   */
  public SVM(List<SVM> svmList) {
    int dim = svmList.get(0).getWeights().getDimension();
    RealVector weights = new RealVector(dim);
    for (SVM svm : svmList)
      weights.add(svm.getWeights());

    this.weights = weights.scaleThis(1.0/svmList.size());
  }

  /**
   * Given a training instance it returns the result of sign(weights'instanceFeatures).
   */
  public int classify(TrainingInstance ti) {
    RealVector features = ti.getFeatures();
    double result = ti.getFeatures().dotProduct(this.weights);
    if (result >= 0) return 1;
    else return -1;
  }

  public RealVector getWeights() {
    return this.weights;
  }

  @Override
  public String toString() {
    return weights.toString();
  }
}
