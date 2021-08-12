package ru.serobyan;

public class Main {

    public static void main(String[] args) {
        try {
            var classifier = ImageClassifier.of("C:\\image\\training", "C:\\image\\testing");
            classifier.buildModel();
            var trainingDataSetIterator = classifier.getTrainingDataSetIterator();
            classifier.trainModel(trainingDataSetIterator);
            var testingDataSetIterator = classifier.getTestingDataSetIterator();
            classifier.evaluatingModel(testingDataSetIterator);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
