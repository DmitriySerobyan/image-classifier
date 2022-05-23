package ru.serobyan;

import lombok.SneakyThrows;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

public class Main {

    private static final String TRAINING_IMAGE_FOLDERS_PATH = "C:/image/training";
    private static final String TESTING_IMAGE_FOLDERS_PATH = "C:/image/testing";
    private static final String IMAGE_FOR_CLASSIFICATION_FOLDER_PATH = "D:/PicturesU";
    private static final String CLASSIFIED_IMAGE_FOLDER_PATH = "D:/classification_test";

    public static void main(String[] args) {
        var classifier = ImageClassifier.of(TRAINING_IMAGE_FOLDERS_PATH, TESTING_IMAGE_FOLDERS_PATH);
        classifier.buildModel();
        classifier.trainModel();
        classifier.evaluatingModel();

        var imageForClassificationFolder = new File(IMAGE_FOR_CLASSIFICATION_FOLDER_PATH);
        var imagesForClassification = imageForClassificationFolder.listFiles();
        for (var image : imagesForClassification) {
            var labelOpt = classifier.predict(image);
            labelOpt.ifPresent((label) -> moveImage(image, label));
        }
    }

    @SneakyThrows
    private static void moveImage(File image, String label) {
        Files.move(
            Paths.get(image.getAbsolutePath()),
            Paths.get(CLASSIFIED_IMAGE_FOLDER_PATH + "/" + label + "/" + image.getName()),
            StandardCopyOption.REPLACE_EXISTING
        );
    }

}
