use ndarray::{arr2, Array2, Axis};

use pyrus_nn::network::Sequential;
use pyrus_nn::layers::{Layer, Dense};
use pyrus_nn::costs;
use pyrus_nn::activations::Activation;


#[test]
fn test_sequential_network_classification() {

    let mut network = Sequential::new();
    assert!(
        network.add(Dense::new(4, 10, Activation::Sigmoid)).is_ok()
    );
    assert!(
        network.add(Dense::new(10, 8, Activation::Sigmoid)).is_ok()
    );
    assert!(
        network.add(Dense::new(8, 6, Activation::Sigmoid)).is_ok()
    );
    assert!(
        network.add(Dense::new(6, 3, Activation::Softmax)).is_ok()
    );

    if let Ok(_) = network.add(Dense::new(5, 2, Activation::Linear)) {
        panic!("Added layer with mismatched sizes!")
    }

    let (x, y) = _get_iris();
    let out = network.predict(x.view());
    println!("Output before back prop: {:#?}", &out);

    network.fit(x.view(), y.view());
    let out = network.predict(x.view());
    //println!("Output: {:?}", &out);
    println!("Output after back prop: {:#?}", &out.as_slice().unwrap()[0..4]);

    for (i, (pred, actual)) in out.axis_iter(Axis(0)).zip(y.axis_iter(Axis(0))).enumerate() {

        if i % 20 == 0 {
            println!("Predicted: {:?} ------Actual -> {:?}", pred, actual);
        }
    }

    let mse = costs::mean_squared_error(y.view(), out.view());
    println!("MSE: {}", mse);
    assert!(mse > 0.6);

    // Array of two predictions
    //assert_eq!(out.shape(), &[1, 2]);
}

//#[test]
fn test_sequential_network_regression() {
    let mut network = Sequential::new();
    assert!(
        network.add(Dense::new(3, 12, Activation::Linear)).is_ok()
    );
    assert!(
        network.add(Dense::new(12, 24, Activation::Linear)).is_ok()
    );
    assert!(
        network.add(Dense::new(24, 1, Activation::Linear)).is_ok()
    );

    let (x, y) = _get_regression();

    network.fit(x.view(), y.view());

    let out = network.predict(x.view());
    println!("Out: {:?}", &out.as_slice().unwrap()[0..4]);
    println!("target: {:?}", &y.as_slice().unwrap()[0..4]);
}


fn _get_iris() -> (Array2<f32>, Array2<f32>) {
    let X = arr2(&[
        [5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5.0,3.6,1.4,0.2],
        [5.4,3.9,1.7,0.4], [4.6,3.4,1.4,0.3], [5.0,3.4,1.5,0.2], [4.4,2.9,1.4,0.2], [4.9,3.1,1.5,0.1],
        [5.4,3.7,1.5,0.2], [4.8,3.4,1.6,0.2], [4.8,3.0,1.4,0.1], [4.3,3.0,1.1,0.1], [5.8,4.0,1.2,0.2],
        [5.7,4.4,1.5,0.4], [5.4,3.9,1.3,0.4], [5.1,3.5,1.4,0.3], [5.7,3.8,1.7,0.3], [5.1,3.8,1.5,0.3],
        [5.4,3.4,1.7,0.2], [5.1,3.7,1.5,0.4], [4.6,3.6,1.0,0.2], [5.1,3.3,1.7,0.5], [4.8,3.4,1.9,0.2],
        [5.0,3.0,1.6,0.2], [5.0,3.4,1.6,0.4], [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.2], [4.7,3.2,1.6,0.2],
        [4.8,3.1,1.6,0.2], [5.4,3.4,1.5,0.4], [5.2,4.1,1.5,0.1], [5.5,4.2,1.4,0.2], [4.9,3.1,1.5,0.1],
        [5.0,3.2,1.2,0.2], [5.5,3.5,1.3,0.2], [4.9,3.1,1.5,0.1], [4.4,3.0,1.3,0.2], [5.1,3.4,1.5,0.2],
        [5.0,3.5,1.3,0.3], [4.5,2.3,1.3,0.3], [4.4,3.2,1.3,0.2], [5.0,3.5,1.6,0.6], [5.1,3.8,1.9,0.4],
        [4.8,3.0,1.4,0.3], [5.1,3.8,1.6,0.2], [4.6,3.2,1.4,0.2], [5.3,3.7,1.5,0.2], [5.0,3.3,1.4,0.2],

        [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4.0,1.3], [6.5,2.8,4.6,1.5],
        [5.7,2.8,4.5,1.3], [6.3,3.3,4.7,1.6], [4.9,2.4,3.3,1.0], [6.6,2.9,4.6,1.3], [5.2,2.7,3.9,1.4],
        [5.0,2.0,3.5,1.0], [5.9,3.0,4.2,1.5], [6.0,2.2,4.0,1.0], [6.1,2.9,4.7,1.4], [5.6,2.9,3.6,1.3],
        [6.7,3.1,4.4,1.4], [5.6,3.0,4.5,1.5], [5.8,2.7,4.1,1.0], [6.2,2.2,4.5,1.5], [5.6,2.5,3.9,1.1],
        [5.9,3.2,4.8,1.8], [6.1,2.8,4.0,1.3], [6.3,2.5,4.9,1.5], [6.1,2.8,4.7,1.2], [6.4,2.9,4.3,1.3],
        [6.6,3.0,4.4,1.4], [6.8,2.8,4.8,1.4], [6.7,3.0,5.0,1.7], [6.0,2.9,4.5,1.5], [5.7,2.6,3.5,1.0],
        [5.5,2.4,3.8,1.1], [5.5,2.4,3.7,1.0], [5.8,2.7,3.9,1.2], [6.0,2.7,5.1,1.6], [5.4,3.0,4.5,1.5],
        [6.0,3.4,4.5,1.6], [6.7,3.1,4.7,1.5], [6.3,2.3,4.4,1.3], [5.6,3.0,4.1,1.3], [5.5,2.5,4.0,1.3],
        [5.5,2.6,4.4,1.2], [6.1,3.0,4.6,1.4], [5.8,2.6,4.0,1.2], [5.0,2.3,3.3,1.0], [5.6,2.7,4.2,1.3],
        [5.7,3.0,4.2,1.2], [5.7,2.9,4.2,1.3], [6.2,2.9,4.3,1.3], [5.1,2.5,3.0,1.1], [5.7,2.8,4.1,1.3],

        [6.3,3.3,6.0,2.5], [5.8,2.7,5.1,1.9], [7.1,3.0,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3.0,5.8,2.2],
        [7.6,3.0,6.6,2.1], [4.9,2.5,4.5,1.7], [7.3,2.9,6.3,1.8], [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5],
        [6.5,3.2,5.1,2.0], [6.4,2.7,5.3,1.9], [6.8,3.0,5.5,2.1], [5.7,2.5,5.0,2.0], [5.8,2.8,5.1,2.4],
        [6.4,3.2,5.3,2.3], [6.5,3.0,5.5,1.8], [7.7,3.8,6.7,2.2], [7.7,2.6,6.9,2.3], [6.0,2.2,5.0,1.5],
        [6.9,3.2,5.7,2.3], [5.6,2.8,4.9,2.0], [7.7,2.8,6.7,2.0], [6.3,2.7,4.9,1.8], [6.7,3.3,5.7,2.1],
        [7.2,3.2,6.0,1.8], [6.2,2.8,4.8,1.8], [6.1,3.0,4.9,1.8], [6.4,2.8,5.6,2.1], [7.2,3.0,5.8,1.6],
        [7.4,2.8,6.1,1.9], [7.9,3.8,6.4,2.0], [6.4,2.8,5.6,2.2], [6.3,2.8,5.1,1.5], [6.1,2.6,5.6,1.4],
        [7.7,3.0,6.1,2.3], [6.3,3.4,5.6,2.4], [6.4,3.1,5.5,1.8], [6.0,3.0,4.8,1.8], [6.9,3.1,5.4,2.1],
        [6.7,3.1,5.6,2.4], [6.9,3.1,5.1,2.3], [5.8,2.7,5.1,1.9], [6.8,3.2,5.9,2.3], [6.7,3.3,5.7,2.5],
        [6.7,3.0,5.2,2.3], [6.3,2.5,5.0,1.9], [6.5,3.0,5.2,2.0], [6.2,3.4,5.4,2.3], [5.9,3.0,5.1,1.8],
    ]);
    let y = arr2(&[
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],

        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],

        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],

    ]);

    (X, y)
}


fn _get_regression() -> (Array2<f32>, Array2<f32>) {

    let x = arr2(&[
       [-0.10196091,  0.02040948,  0.69330399],
       [-0.26010187,  0.86505589, -0.96724357],
       [ 1.43009105,  1.13701358,  1.70439351],
       [ 1.07446856,  1.48033539, -1.01673459],
       [-0.35705518, -0.81186403, -2.28900087],
       [-1.13659922,  1.03712764, -2.72237136],
       [ 0.18886966,  0.82715883, -1.92723941],
       [-0.15276682, -0.55774752,  0.84560644],
       [ 1.36400812,  0.82757557, -2.18240516],
       [-0.6580276 , -1.02912292,  0.23295469],
       [ 1.33207777,  0.82292028, -0.68718441],
       [ 0.34793648,  0.01246602,  0.25493871],
       [-1.78972059, -0.45329447,  1.32982866],
       [-0.17860166, -0.59500509, -0.67615077],
       [ 0.27571946,  0.69765649,  0.28452654],
       [ 1.18873376,  1.69331441,  0.21828912],
       [-0.19375227, -0.13084449, -0.91757923],
       [ 0.86504129,  0.69524635, -0.55961809],
       [ 1.65771471,  0.58963461, -1.46214721],
       [-0.04416794, -0.11891432, -0.0908873 ],
       [-0.11761116,  0.44455161, -0.32250362],
       [-0.07763712, -0.32971857,  0.28508845],
       [ 0.86026789,  2.22162551, -0.08523199],
       [ 0.74230538, -0.200106  , -0.48383539],
       [-0.84445467, -0.04938064, -2.30397345],
       [ 0.86444357, -0.5969149 ,  1.37176954],
       [ 2.99019197,  0.37895551, -2.21683412],
       [-0.65538555,  2.43405309, -0.57676173],
       [ 0.20404891, -1.35713437,  0.36170782],
       [ 0.33336406,  0.52147131, -1.33722359],
       [ 0.05016553, -1.56092891,  0.52461879],
       [ 0.74102077, -0.27420627, -0.62610206],
       [ 0.27034476, -0.49608932,  0.66338511],
       [ 0.88169513,  0.8476385 , -0.08288048],
       [ 1.24821995, -0.29992286, -0.20560647],
       [-1.47845635,  0.44144689,  0.29111265],
       [ 0.15336749,  1.12437172, -1.58273542],
       [ 0.96488213, -0.47609018,  0.49050603],
       [-0.59502221,  0.43348189, -0.41318518],
       [-1.40004367, -0.33818332,  1.09842095],
       [ 0.14025203,  0.48066287, -1.22021405],
       [-0.8317773 , -0.14306914, -0.61831617],
       [ 0.66760113, -1.68008605, -0.68899594],
       [ 1.03688324,  0.05588467, -0.69007793],
       [-0.07742534,  0.67101061,  0.08172535],
       [ 0.15903678, -0.00990032, -0.58520939],
       [ 1.66456444, -0.07293532,  2.08930765],
       [ 0.34795312, -0.33930859, -0.26544627],
       [-0.28283884, -0.55224697, -0.16798201],
       [ 0.53161657, -0.56654647, -0.01569703],
       [-0.91112647,  0.1173078 , -1.84016765],
       [ 0.69675163,  0.49089609, -1.84917332],
       [-1.28647979, -0.72490559,  1.31976532],
       [-0.37579136, -0.9567898 , -0.99735311],
       [ 1.35057163, -1.08621411, -0.61863176],
       [-0.12672578,  1.08126348,  0.70097846],
       [-0.26062027, -0.30178222, -0.43280725],
       [-1.04739266,  1.13430932,  0.42620378],
       [-0.21923991, -0.08823843,  1.01775857],
       [ 0.74604489,  0.2445787 ,  0.21177138],
       [-1.96339073,  1.29444931,  1.10808965],
       [-1.10780801, -0.77704456, -1.18404821],
       [ 1.09276022,  0.9570646 , -0.36816731],
       [ 0.03962253,  0.10183657, -0.80315596],
       [ 0.83908502,  0.25030715,  1.45116789],
       [ 0.08056916,  0.19174203,  0.59261759],
       [ 0.48703447,  1.16055061, -0.02814975],
       [ 0.30250728, -0.24200004, -0.4845105 ],
       [-1.10515555, -0.84367459, -0.61095385],
       [-0.23663847,  0.28888729, -0.35656868],
       [-1.40506743,  0.87841158,  0.51184455],
       [-0.43972128, -1.20725731, -0.98634687],
       [ 0.06238904,  0.20621189, -1.61347283],
       [ 0.98422163, -0.26024588,  1.29483184],
       [ 1.45621682,  0.69945208, -0.75777326],
       [ 1.21617406,  0.30171454, -1.48973408],
       [-1.00317481,  0.98167343,  0.66347554],
       [-0.93302485,  1.33009879,  1.60218169],
       [ 0.64026915, -0.76163067, -0.79189589],
       [-0.18320636,  1.39122517, -0.53657891],
       [ 0.48222199, -0.51528263,  0.26560743],
       [ 2.54916671, -0.86445242,  0.12315656],
       [-0.59008164,  1.9492768 ,  0.73457853],
       [ 2.27732073, -0.37294806,  0.54365016],
       [ 1.2717979 ,  0.17935289,  0.52575321],
       [ 2.20982893,  0.77978846,  0.83946136],
       [-0.19583997, -0.75798258, -0.46826873],
       [-0.72153485, -0.13424402, -1.36067203],
       [-0.20619838,  1.20436544,  3.06385083],
       [ 0.90703806, -0.39230899,  0.07948281],
       [-1.13695395, -0.33265079,  0.24818223],
       [-0.52566243, -0.2922805 , -0.0276154 ],
       [-0.86462379,  1.09476317, -1.30908416],
       [-1.03061688,  1.34788054,  1.38787769],
       [-0.48624911,  0.01244013,  0.7691549 ],
       [ 2.36759494, -0.60457339, -1.19619902],
       [-1.91193255, -0.06514105,  0.9099901 ],
       [ 0.89275688, -0.53559351,  0.53623464],
       [ 0.1124365 , -0.86477438, -2.2037671 ],
       [-1.85844779, -0.13298393, -1.94611922],
       [ 0.25970365, -1.25193112, -1.48336593],
       [-0.34652638, -0.44026387,  1.40927662],
       [-0.39691771, -0.40975378,  0.99932547],
       [-0.65372046, -0.74912369, -0.13845833],
       [ 0.64762109,  0.07745077,  0.27279514],
       [-1.76661602,  1.34047933,  0.39316247],
       [ 0.38289823,  0.22396543, -1.11386587],
       [ 0.66185455,  0.29711089, -0.83298395],
       [ 0.16778673, -0.22365267,  0.71760161],
       [ 0.12758968,  1.51989882, -1.00157976],
       [ 2.02156235, -0.241601  ,  1.06457183],
       [-0.48805336,  1.42902189, -0.00982557],
       [-0.19841681,  0.10462078, -0.61294396],
       [ 1.21944496, -1.82383096, -0.48650621],
       [ 1.04721274, -0.20332815, -0.258142  ],
       [-0.5806038 , -0.58768598, -1.07215421],
       [ 0.26728816,  1.00557676,  1.01290503],
       [-0.17223185, -0.97606695, -2.52567288],
       [ 0.66638575,  0.06953358, -1.15274738],
       [-0.19737086,  0.01772197,  1.05478463],
       [-0.70471919, -1.52183277,  0.06032627],
       [-0.39204038, -1.06747647, -0.57488137],
       [-0.34408154, -0.08657502,  1.26318765],
       [-0.22726904,  0.64628987,  0.02227798],
       [-0.32175596,  0.32080905, -0.33608797],
       [ 0.4327195 ,  0.93525524,  0.85246792],
       [-0.68943412,  1.77257747,  0.20842327],
       [-1.10475305, -1.56976665, -1.31929198],
       [-0.71169322,  0.27549333,  0.794103  ],
       [-2.42145371,  0.23969877,  1.87827203],
       [ 0.47138396, -0.38023119, -0.38411232],
       [-0.11797158,  0.28652018, -2.1888354 ],
       [-2.07286267,  0.52138768, -0.44851461],
       [-0.41620692,  1.61992902, -0.43548056],
       [ 1.20918668, -0.5439003 , -1.78609959],
       [-0.54088537,  0.21222435, -1.50949721],
       [-0.08945656, -0.35398853,  0.23370088],
       [ 0.12889402, -1.62645129,  1.03060441],
       [ 0.82407613, -0.44243247, -0.20471126],
       [-1.57815036, -0.8788196 , -0.7976027 ],
       [ 1.09929818, -0.34493992, -0.80391792],
       [-0.69482869, -0.05635361,  0.87590891],
       [ 1.17880643,  0.56503572, -0.33767493],
       [ 0.09232921,  0.74489643, -0.38931606],
       [-1.24159383,  1.06336073,  0.37495778],
       [ 0.61267159, -1.04691845,  0.08922983],
       [ 1.14323044,  1.06047815,  1.73383048],
       [-1.19248413, -1.32601856, -0.71077803],
       [ 0.15676421, -0.23314754, -1.0401304 ],
       [ 0.82146329,  0.04573372, -0.51953585]]);

    let y = arr2(&[
       [ 5.48111697e+01],
       [-6.04182363e+01],
       [ 1.97808987e+02],
       [-2.10506630e+01],
       [-2.14980211e+02],
       [-2.15899139e+02],
       [-1.30398315e+02],
       [ 5.05523971e+01],
       [-1.27532585e+02],
       [-2.20664443e+01],
       [-6.86629063e+00],
       [ 2.80219419e+01],
       [ 5.98547261e+01],
       [-7.45595207e+01],
       [ 4.74740226e+01],
       [ 8.72927043e+01],
       [-8.19398946e+01],
       [-9.32336775e+00],
       [-6.95685603e+01],
       [-1.14777999e+01],
       [-1.65499052e+01],
       [ 1.26940441e+01],
       [ 7.03151963e+01],
       [-2.98030903e+01],
       [-2.05390418e+02],
       [ 1.12635587e+02],
       [-1.09813030e+02],
       [ 5.72167096e+00],
       [-3.16889710e+00],
       [-8.78344342e+01],
       [ 1.47244179e+00],
       [-4.33833697e+01],
       [ 4.59072164e+01],
       [ 3.38425837e+01],
       [ 2.48643810e-01],
       [ 5.89699879e+00],
       [-9.51082763e+01],
       [ 4.63418562e+01],
       [-3.37919203e+01],
       [ 5.19871804e+01],
       [-8.33078611e+01],
       [-7.07666796e+01],
       [-8.79187854e+01],
       [-3.37315813e+01],
       [ 2.31990579e+01],
       [-4.46035323e+01],
       [ 2.01108751e+02],
       [-2.37370345e+01],
       [-3.42262175e+01],
       [-5.90373315e+00],
       [-1.64560561e+02],
       [-1.22946423e+02],
       [ 6.18015699e+01],
       [-1.14369016e+02],
       [-5.24686699e+01],
       [ 8.35774213e+01],
       [-4.85261157e+01],
       [ 4.42221329e+01],
       [ 7.58751703e+01],
       [ 3.87699722e+01],
       [ 8.55458059e+01],
       [-1.39365528e+02],
       [ 1.78622386e+01],
       [-6.16831264e+01],
       [ 1.41447482e+02],
       [ 5.49217060e+01],
       [ 3.88168566e+01],
       [-3.98128497e+01],
       [-9.45679296e+01],
       [-2.59070381e+01],
       [ 3.70926510e+01],
       [-1.21519685e+02],
       [-1.24217470e+02],
       [ 1.17879402e+02],
       [-1.34416778e+01],
       [-8.84408466e+01],
       [ 6.02583424e+01],
       [ 1.47308157e+02],
       [-7.20285439e+01],
       [-9.69453068e+00],
       [ 1.73350176e+01],
       [ 3.78097907e+01],
       [ 1.00443376e+02],
       [ 7.97746196e+01],
       [ 7.30572212e+01],
       [ 1.33564856e+02],
       [-6.24223845e+01],
       [-1.28606137e+02],
       [ 2.77204223e+02],
       [ 1.40623655e+01],
       [-1.16360864e+01],
       [-2.06801057e+01],
       [-9.41078853e+01],
       [ 1.28425717e+02],
       [ 5.30459724e+01],
       [-6.59671637e+01],
       [ 3.37849882e+01],
       [ 4.70021526e+01],
       [-2.00066791e+02],
       [-1.98929058e+02],
       [-1.49057607e+02],
       [ 9.56144093e+01],
       [ 6.21333546e+01],
       [-4.45847700e+01],
       [ 3.72392641e+01],
       [ 3.26745022e+01],
       [-7.67326241e+01],
       [-4.63495616e+01],
       [ 5.56076322e+01],
       [-3.77495561e+01],
       [ 1.20495214e+02],
       [ 2.79892074e+01],
       [-5.09360329e+01],
       [-6.42825407e+01],
       [-5.44312413e+00],
       [-1.14588482e+02],
       [ 1.14772187e+02],
       [-2.34926019e+02],
       [-7.83717631e+01],
       [ 8.21815799e+01],
       [-5.03247360e+01],
       [-8.33726850e+01],
       [ 9.33475457e+01],
       [ 1.46973247e+01],
       [-2.50896899e+01],
       [ 1.03163148e+02],
       [ 5.09485777e+01],
       [-1.71688999e+02],
       [ 5.76505316e+01],
       [ 1.10429818e+02],
       [-3.20026833e+01],
       [-1.72395467e+02],
       [-6.39387184e+01],
       [ 1.55444791e-02],
       [-1.35479062e+02],
       [-1.27714730e+02],
       [ 7.62830659e+00],
       [ 4.23761424e+01],
       [-1.20358187e+01],
       [-1.20165170e+02],
       [-5.25457491e+01],
       [ 5.56738022e+01],
       [ 1.14814413e+01],
       [-9.65544552e+00],
       [ 3.42484832e+01],
       [-8.72455904e+00],
       [ 1.92378004e+02],
       [-1.17448985e+02],
       [-8.76220068e+01],
       [-2.44773428e+01]]);

    (x, y)
}