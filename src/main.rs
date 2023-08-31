use std::collections::HashMap;

// Enumeration for labels
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Label {
    A,
    B,
}

// Struct to hold data points with features and labels
struct DataPoint {
    features: Vec<f64>,
    label: Label,
}

impl DataPoint {
    fn new(features: Vec<f64>, label: Label) -> DataPoint {
        DataPoint { features, label }
    }
}

// K-nearest neighbors (KNN) algorithm implementation
struct KNN {
    training_data: Vec<DataPoint>,
}

// Trait for distance calculation
trait DistanceMetric {
    fn calculate(&self, vec1: &[f64], vec2: &[f64]) -> f64;
}

// Implementation of Euclidean distance as a distance metric
struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn calculate(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl KNN {
    // Constructor for KNN
    fn new(training_data: Vec<DataPoint>) -> KNN {
        KNN { training_data }
    }

    // Classification method to predict the label of an input based on KNN
    fn classify(&self, input_features: Vec<f64>, k: usize, distance_metric: &dyn DistanceMetric) -> Option<Label> {
        let mut distances: Vec<(f64, &DataPoint)> = self
            .training_data
            .iter()
            .map(|data_point| {
                let distance = distance_metric.calculate(&data_point.features, &input_features);
                (distance, data_point)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let nearest_neighbors = distances.iter().take(k);

        let mut label_counts = HashMap::new();
        for (_, data_point) in nearest_neighbors {
            *label_counts.entry(data_point.label).or_insert(0) += 1;
        }

        let most_common_label = label_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label);

        most_common_label
    }
}

fn main() {
    // Sample training data
    let training_data = vec![
        DataPoint::new(vec![2.0, 3.0], Label::A),
        DataPoint::new(vec![5.0, 8.0], Label::B),
        DataPoint::new(vec![1.0, 1.0], Label::A),
    ];

    // Create KNN instance with training data
    let knn = KNN::new(training_data);

    // Input features to predict
    let input_features = vec![3.0, 4.0];
    let k = 2;

    // Instantiate Euclidean distance metric
    let euclidean_distance = EuclideanDistance;

    // Classify the input features using KNN
    let predicted_label = knn
        .classify(input_features.clone(), k, &euclidean_distance)
        .unwrap_or(Label::A);

    // Print the predicted label
    println!("Predicted Label: {:?}", predicted_label);
}
