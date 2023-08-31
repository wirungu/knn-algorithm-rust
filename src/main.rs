use std::collections::HashMap;

// Holds data points and their labels
struct DataPoint {
    features: Vec<f64>,
    label: String,
}

impl DataPoint {
    fn new(features: Vec<f64>, label: &str) -> DataPoint {
        DataPoint {
            features,
            label: String::from(label),
        }
    }
}

struct KNN {
    training_data: Vec<DataPoint>,
}

impl KNN {
    fn new(training_data: Vec<DataPoint>) -> KNN {
        KNN { training_data }
    }

    fn classify(&self, input_features: Vec<f64>, k: usize) -> Option<String> {
        // First find distances between input features and training data
        let mut distances: Vec<(f64, &DataPoint)> = self
            .training_data
            .iter()
            .map(|data_point| {
                let distance = euclidean_distance(&data_point.features, &input_features);
                (distance, data_point)
            })
            .collect();



        // Then sort distances in ascending order
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find k nearest neighbors
        let nearest_neighbors = distances.iter().take(k);

        // Count how many times each label occurs in these neighbors
        let mut label_counts = HashMap::new();
        for (_, data_point) in nearest_neighbors {
            *label_counts.entry(&data_point.label).or_insert(0) += 1;
        }

        // Find most common label
        let most_common_label = label_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label.to_string());

        most_common_label
    }
}

// Euclidean distance function. TODO: Add other functions
fn euclidean_distance(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Create sample data, initialize KNN and classify
fn main() {
    let training_data = vec![
        DataPoint::new(vec![2.0, 3.0], "A"),
        DataPoint::new(vec![5.0, 8.0], "B"),
        DataPoint::new(vec![1.0, 1.0], "A"),
    ];

    let knn = KNN::new(training_data);

    let input_features = vec![3.0, 4.0];
    let k = 2;

    let predicted_label = knn
        .classify(input_features.clone(), k)
        .unwrap_or("Unknown".to_string());

    println!("Predicted Label: {}", predicted_label);
}
