import Foundation

// Runtime subset of the measured frozen D quality record, with local paths and
// performance claims omitted. Original SHA-256: 9180092aebe2ea6b5cceba0e43f93ff91888ebda75331979ff4c9bdc1d364b2e.
// Swift parity evidence is in results/swift-d-pipeline-v1.json. Model identity
// and all supported kernel settings are checked by PrivateGDNProfile.validate.
enum FrozenPrivateDProfile {
    static let data = Data(#"""
{
  "package_fingerprint": "2d481d864704eaabf57a57e77544a6439dd8a524752f74ad5b8543b7f7fe3ecb",
  "settings": {
    "mlp": true,
    "recurrence_ane": true,
    "sequence_length": 2112,
    "fraction": 0.75,
    "max_layers": 24,
    "recurrence_layer_slots": [
      0
    ]
  },
  "quality_gate": {
    "minimum_vector_cosine": 0.999,
    "reference": {
      "minimum_cosine": 0.999,
      "maximum_relative_l2": 0.05,
      "minimum_output_cosine": 0.9999969915557261,
      "minimum_state_cosine": 0.9999922796003613,
      "maximum_output_relative_l2": 0.0024530718725297106,
      "maximum_state_relative_l2": 0.0039294754588061745,
      "nonfinite": 0,
      "passed": true
    },
    "passed": true
  },
  "comparison": {
    "vector_cosine": 0.9997633695602417
  },
  "model": {
    "recurrence": {
      "algorithm": "block-forward-substitution-v1",
      "solve_block_size": 8,
      "query_scale": 4096,
      "max_tokens": 8192,
      "io_dtype": "fp16",
      "enabled_layer_slots": [
        0
      ]
    }
  }
}
"""#.utf8)
}
