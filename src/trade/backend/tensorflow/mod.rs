use super::*;
use std::env;
use std::fs::File;
use std::io::Read;

pub mod resolvers;

use self::resolvers::Resolver;

pub struct Tensorflow<R: Resolver> {
    session: tf::Session,
    graph: tf::Graph,
    input: tf::Operation,
    output: tf::Operation,
    resolver: R
}

impl<R: Resolver> Tensorflow<R> {
    pub fn new(model: &str, resolver: R) -> Self {
        env::set_var("TF_CPP_MIN_LOG_LEVEL", "3"); //Reduce tensorflow logging

        tf::Library::load("res/lib/_cudnn_rnn_ops.so").unwrap();

        println!("Tensorflow v{}", tf::version().unwrap());

        //Load graph
        let model = format!("res/models/trained/{}.pb", model);
        println!("Loading model at: {}", &model);

        let mut graph = tf::Graph::new();
        let mut proto = Vec::new();
        File::open(&model).unwrap().read_to_end(&mut proto).expect("Faild to open model file.");
        graph.import_graph_def(&proto, &tf::ImportGraphDefOptions::new()).expect("Failed to import graph def");

        //Find the input and output nodes
        println!("Finding input and output nodes");
        let mut input: Option<tf::Operation> = None;
        let mut output: Option<tf::Operation> = None;
        {
            let mut iter = graph.operation_iter();
            while let Some(op) = iter.next() {
                let name = op.name().unwrap();
                if name.contains("input") {
                    input = Some(op);
                } else if name.contains("output") {
                    output = Some(op);
                }
            }
        }

        println!("Allowing for gpu memory growth");
        let mut ses_ops: tf::SessionOptions = tf::SessionOptions::new();
        ses_ops.set_config(&vec![50, 2, 32, 1]); //Allow for gpu memory growth.

        Tensorflow {
            session: tf::Session::new(&ses_ops, &graph).expect("Failed to create session."),
            graph,
            input: input.expect("Expected to have an input node."),
            output: output.expect("Expected to have an output node."),
            resolver
        }
    }
}

impl<T: tf::TensorType + Num + PartialOrd, R: Resolver> Backend<tf::Tensor<T>> for Tensorflow<R> {
    fn evaluate(&mut self, input: &tf::Tensor<T>) -> Action {
        let mut step = tf::StepWithGraph::new();
        step.add_input(&self.input, 0, &input);
        let output_token = step.request_output(&self.output, 0);
        self.session.run(&mut step).unwrap();
        self.resolver.evaluate(&step.take_output::<T>(output_token).unwrap())
    }
}