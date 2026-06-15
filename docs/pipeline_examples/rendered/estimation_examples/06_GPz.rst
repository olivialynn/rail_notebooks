GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4656184e-01	 3.2109187e-01	-3.3632137e-01	 3.1864987e-01	[-3.3219341e-01]	 5.0220585e-01


.. parsed-literal::

       2	-2.7508094e-01	 3.1042854e-01	-2.5026673e-01	 3.1018085e-01	[-2.4984109e-01]	 2.5775647e-01


.. parsed-literal::

       3	-2.3297455e-01	 2.9066340e-01	-1.9078603e-01	 2.9248772e-01	[-1.9994280e-01]	 3.0479455e-01


.. parsed-literal::

       4	-1.8498355e-01	 2.6622178e-01	-1.4048760e-01	 2.6901498e-01	[-1.5887298e-01]	 2.0583081e-01


.. parsed-literal::

       5	-1.0135612e-01	 2.5502930e-01	-6.9063079e-02	 2.5942239e-01	[-9.3036048e-02]	 2.1024656e-01


.. parsed-literal::

       6	-6.7047172e-02	 2.5074374e-01	-3.7127140e-02	 2.5386968e-01	[-4.8755817e-02]	 2.1217179e-01
       7	-4.9095069e-02	 2.4765502e-01	-2.3915782e-02	 2.5084318e-01	[-3.6428696e-02]	 2.0135188e-01


.. parsed-literal::

       8	-3.5604525e-02	 2.4550355e-01	-1.4647186e-02	 2.4881202e-01	[-2.8871819e-02]	 2.1560669e-01


.. parsed-literal::

       9	-2.4816900e-02	 2.4338108e-01	-5.9179664e-03	 2.4739207e-01	[-2.2188053e-02]	 2.1173811e-01


.. parsed-literal::

      10	-1.5148703e-02	 2.4169350e-01	 1.0998601e-03	 2.4612110e-01	[-1.8545046e-02]	 2.1736193e-01


.. parsed-literal::

      11	-9.9822881e-03	 2.4085586e-01	 5.6502681e-03	 2.4467127e-01	[-1.0282316e-02]	 2.1918607e-01
      12	-6.0013755e-03	 2.4021955e-01	 8.8476090e-03	 2.4339864e-01	[-3.8426771e-03]	 1.9220853e-01


.. parsed-literal::

      13	-2.0870499e-03	 2.3950015e-01	 1.2487733e-02	 2.4242444e-01	[ 1.3904995e-03]	 2.0919275e-01


.. parsed-literal::

      14	 1.2412242e-01	 2.2586221e-01	 1.4821022e-01	 2.2782005e-01	[ 1.4064375e-01]	 4.7061849e-01
      15	 1.7537189e-01	 2.2321812e-01	 1.9969407e-01	 2.2614068e-01	[ 1.9096703e-01]	 2.0223594e-01


.. parsed-literal::

      16	 2.7193949e-01	 2.1813178e-01	 3.0078777e-01	 2.2074339e-01	[ 2.9081852e-01]	 2.0857358e-01


.. parsed-literal::

      17	 3.5966046e-01	 2.1214135e-01	 3.9281866e-01	 2.1327991e-01	[ 3.8715550e-01]	 2.1117997e-01


.. parsed-literal::

      18	 4.3435225e-01	 2.0935105e-01	 4.7087572e-01	 2.0834566e-01	[ 4.6868318e-01]	 2.1081018e-01


.. parsed-literal::

      19	 5.0942423e-01	 2.0537579e-01	 5.4717681e-01	 2.0591370e-01	[ 5.3875954e-01]	 2.1570897e-01


.. parsed-literal::

      20	 5.8655835e-01	 1.9734206e-01	 6.2609245e-01	 1.9782356e-01	[ 6.1930395e-01]	 2.1629381e-01


.. parsed-literal::

      21	 6.5430776e-01	 1.8933368e-01	 6.9517015e-01	 1.9016991e-01	[ 6.9282464e-01]	 2.1650791e-01


.. parsed-literal::

      22	 6.9565279e-01	 1.8654618e-01	 7.3722820e-01	 1.8392292e-01	[ 7.3045933e-01]	 2.0683265e-01


.. parsed-literal::

      23	 7.2898648e-01	 1.8596664e-01	 7.6811289e-01	 1.8372829e-01	[ 7.6472188e-01]	 2.1394897e-01


.. parsed-literal::

      24	 7.5255510e-01	 1.9218262e-01	 7.9268567e-01	 1.8864951e-01	[ 7.9374824e-01]	 2.0134330e-01
      25	 7.7982227e-01	 1.8935930e-01	 8.2026958e-01	 1.8454323e-01	[ 8.2057536e-01]	 2.0755363e-01


.. parsed-literal::

      26	 8.0309680e-01	 1.8443206e-01	 8.4360052e-01	 1.7957010e-01	[ 8.4701569e-01]	 2.1275496e-01


.. parsed-literal::

      27	 8.2541246e-01	 1.8209748e-01	 8.6611648e-01	 1.7673161e-01	[ 8.7001999e-01]	 2.0278192e-01


.. parsed-literal::

      28	 8.5807957e-01	 1.7676277e-01	 8.9911346e-01	 1.7223213e-01	[ 9.0112267e-01]	 2.0460224e-01


.. parsed-literal::

      29	 8.8083715e-01	 1.7631602e-01	 9.2338990e-01	 1.6992358e-01	[ 9.3033434e-01]	 2.0599055e-01


.. parsed-literal::

      30	 9.0262498e-01	 1.7462610e-01	 9.4606821e-01	 1.6949943e-01	[ 9.5362212e-01]	 2.2023058e-01


.. parsed-literal::

      31	 9.1395313e-01	 1.7329186e-01	 9.5709985e-01	 1.6776213e-01	[ 9.6311483e-01]	 2.0890188e-01
      32	 9.2626768e-01	 1.7296666e-01	 9.6949409e-01	 1.6706365e-01	[ 9.7378059e-01]	 1.9569707e-01


.. parsed-literal::

      33	 9.4001270e-01	 1.7194425e-01	 9.8415881e-01	 1.6625539e-01	[ 9.8224513e-01]	 2.3814416e-01


.. parsed-literal::

      34	 9.5566811e-01	 1.7171990e-01	 1.0005488e+00	 1.6626393e-01	[ 9.9736258e-01]	 2.1571970e-01


.. parsed-literal::

      35	 9.6831683e-01	 1.7088328e-01	 1.0135699e+00	 1.6619624e-01	[ 1.0089248e+00]	 2.1359515e-01


.. parsed-literal::

      36	 9.8305322e-01	 1.6981742e-01	 1.0291072e+00	 1.6576582e-01	[ 1.0275221e+00]	 2.1018553e-01


.. parsed-literal::

      37	 9.9516910e-01	 1.6926576e-01	 1.0414173e+00	 1.6512593e-01	[ 1.0374206e+00]	 2.3793530e-01


.. parsed-literal::

      38	 1.0044288e+00	 1.6866680e-01	 1.0510262e+00	 1.6437445e-01	[ 1.0490276e+00]	 2.1401906e-01


.. parsed-literal::

      39	 1.0196873e+00	 1.6850952e-01	 1.0672552e+00	 1.6429172e-01	[ 1.0653807e+00]	 2.1324062e-01


.. parsed-literal::

      40	 1.0302546e+00	 1.7038298e-01	 1.0789779e+00	 1.6637515e-01	[ 1.0764780e+00]	 2.1579266e-01


.. parsed-literal::

      41	 1.0408414e+00	 1.6849057e-01	 1.0894937e+00	 1.6505191e-01	[ 1.0843815e+00]	 2.2077966e-01


.. parsed-literal::

      42	 1.0493089e+00	 1.6681910e-01	 1.0979043e+00	 1.6380116e-01	[ 1.0909468e+00]	 2.1982551e-01


.. parsed-literal::

      43	 1.0597998e+00	 1.6527290e-01	 1.1084437e+00	 1.6275941e-01	[ 1.1001665e+00]	 2.1197653e-01


.. parsed-literal::

      44	 1.0767538e+00	 1.6352506e-01	 1.1262057e+00	 1.6112769e-01	[ 1.1169104e+00]	 2.1086025e-01
      45	 1.0832450e+00	 1.6255821e-01	 1.1331197e+00	 1.6021384e-01	[ 1.1229032e+00]	 2.0529056e-01


.. parsed-literal::

      46	 1.0920359e+00	 1.6206735e-01	 1.1413560e+00	 1.5979904e-01	[ 1.1286953e+00]	 2.1919775e-01


.. parsed-literal::

      47	 1.0989733e+00	 1.6197811e-01	 1.1484122e+00	 1.5944119e-01	[ 1.1336473e+00]	 2.1814585e-01


.. parsed-literal::

      48	 1.1067312e+00	 1.6155317e-01	 1.1562058e+00	 1.5854731e-01	[ 1.1400995e+00]	 2.0902586e-01


.. parsed-literal::

      49	 1.1145639e+00	 1.6149056e-01	 1.1642836e+00	 1.5908211e-01	[ 1.1448032e+00]	 2.1522021e-01
      50	 1.1239755e+00	 1.5998284e-01	 1.1734243e+00	 1.5751501e-01	[ 1.1528356e+00]	 1.9760895e-01


.. parsed-literal::

      51	 1.1322519e+00	 1.5831489e-01	 1.1816504e+00	 1.5606174e-01	[ 1.1618129e+00]	 2.1480227e-01


.. parsed-literal::

      52	 1.1433684e+00	 1.5631103e-01	 1.1928954e+00	 1.5424366e-01	[ 1.1717233e+00]	 2.1151376e-01


.. parsed-literal::

      53	 1.1501856e+00	 1.5441684e-01	 1.2003334e+00	 1.5237955e-01	[ 1.1790031e+00]	 2.2138095e-01
      54	 1.1645134e+00	 1.5294506e-01	 1.2144761e+00	 1.5084853e-01	[ 1.1873588e+00]	 1.9627452e-01


.. parsed-literal::

      55	 1.1701905e+00	 1.5257937e-01	 1.2203428e+00	 1.5016973e-01	[ 1.1902465e+00]	 2.1580005e-01


.. parsed-literal::

      56	 1.1786645e+00	 1.5220473e-01	 1.2292271e+00	 1.4909045e-01	[ 1.1949776e+00]	 2.1836019e-01


.. parsed-literal::

      57	 1.1847341e+00	 1.5224139e-01	 1.2360178e+00	 1.4812392e-01	[ 1.2031257e+00]	 2.1290231e-01


.. parsed-literal::

      58	 1.1934615e+00	 1.5207233e-01	 1.2448173e+00	 1.4780570e-01	[ 1.2090877e+00]	 2.4048662e-01


.. parsed-literal::

      59	 1.1984160e+00	 1.5189005e-01	 1.2497963e+00	 1.4768800e-01	[ 1.2145447e+00]	 2.1846557e-01


.. parsed-literal::

      60	 1.2061802e+00	 1.5179833e-01	 1.2578815e+00	 1.4744742e-01	[ 1.2203875e+00]	 2.1056008e-01


.. parsed-literal::

      61	 1.2130049e+00	 1.5163420e-01	 1.2654645e+00	 1.4731574e-01	[ 1.2225175e+00]	 2.0404339e-01


.. parsed-literal::

      62	 1.2212537e+00	 1.5169040e-01	 1.2738195e+00	 1.4712833e-01	[ 1.2247612e+00]	 2.1631527e-01


.. parsed-literal::

      63	 1.2277519e+00	 1.5110202e-01	 1.2805463e+00	 1.4607254e-01	[ 1.2259767e+00]	 2.2083592e-01


.. parsed-literal::

      64	 1.2342525e+00	 1.5066651e-01	 1.2871977e+00	 1.4584486e-01	[ 1.2280062e+00]	 2.1794558e-01


.. parsed-literal::

      65	 1.2402083e+00	 1.4944203e-01	 1.2933793e+00	 1.4453411e-01	[ 1.2350639e+00]	 2.0307088e-01


.. parsed-literal::

      66	 1.2462142e+00	 1.4916079e-01	 1.2994036e+00	 1.4452983e-01	[ 1.2396061e+00]	 2.2059345e-01
      67	 1.2547061e+00	 1.4871922e-01	 1.3079884e+00	 1.4409831e-01	[ 1.2470262e+00]	 1.8503118e-01


.. parsed-literal::

      68	 1.2624611e+00	 1.4780255e-01	 1.3157842e+00	 1.4286308e-01	[ 1.2517538e+00]	 2.0818329e-01


.. parsed-literal::

      69	 1.2705827e+00	 1.4673218e-01	 1.3239132e+00	 1.4118047e-01	[ 1.2584469e+00]	 2.1000075e-01


.. parsed-literal::

      70	 1.2767076e+00	 1.4608004e-01	 1.3300924e+00	 1.4038624e-01	[ 1.2622515e+00]	 2.1955109e-01


.. parsed-literal::

      71	 1.2835257e+00	 1.4506123e-01	 1.3372636e+00	 1.3931451e-01	[ 1.2637347e+00]	 2.2334313e-01


.. parsed-literal::

      72	 1.2900820e+00	 1.4454493e-01	 1.3439797e+00	 1.3930667e-01	  1.2620822e+00 	 2.1421957e-01


.. parsed-literal::

      73	 1.2962737e+00	 1.4441949e-01	 1.3502401e+00	 1.3956997e-01	[ 1.2693661e+00]	 2.1106291e-01


.. parsed-literal::

      74	 1.3014760e+00	 1.4406617e-01	 1.3554234e+00	 1.3930939e-01	[ 1.2756898e+00]	 2.1646023e-01


.. parsed-literal::

      75	 1.3080946e+00	 1.4326684e-01	 1.3622143e+00	 1.3871883e-01	[ 1.2799904e+00]	 2.0664811e-01


.. parsed-literal::

      76	 1.3130347e+00	 1.4228923e-01	 1.3673234e+00	 1.3800937e-01	[ 1.2849219e+00]	 2.1808839e-01


.. parsed-literal::

      77	 1.3205567e+00	 1.4152563e-01	 1.3747294e+00	 1.3748083e-01	[ 1.2920110e+00]	 2.0475268e-01


.. parsed-literal::

      78	 1.3250258e+00	 1.4098344e-01	 1.3792406e+00	 1.3720118e-01	[ 1.2921361e+00]	 2.0891786e-01


.. parsed-literal::

      79	 1.3295102e+00	 1.4028629e-01	 1.3838691e+00	 1.3677971e-01	  1.2917965e+00 	 2.1228385e-01


.. parsed-literal::

      80	 1.3355157e+00	 1.3943623e-01	 1.3902727e+00	 1.3674785e-01	  1.2862183e+00 	 2.2206426e-01


.. parsed-literal::

      81	 1.3410523e+00	 1.3879175e-01	 1.3959349e+00	 1.3653167e-01	  1.2869219e+00 	 2.1531320e-01
      82	 1.3457567e+00	 1.3833453e-01	 1.4007992e+00	 1.3662382e-01	  1.2852053e+00 	 2.0347404e-01


.. parsed-literal::

      83	 1.3495174e+00	 1.3803739e-01	 1.4047002e+00	 1.3675242e-01	  1.2846159e+00 	 2.1485877e-01


.. parsed-literal::

      84	 1.3546159e+00	 1.3767757e-01	 1.4099711e+00	 1.3689877e-01	  1.2806262e+00 	 2.0950389e-01


.. parsed-literal::

      85	 1.3605974e+00	 1.3716017e-01	 1.4162070e+00	 1.3657733e-01	  1.2790909e+00 	 2.1415830e-01


.. parsed-literal::

      86	 1.3651318e+00	 1.3666092e-01	 1.4212125e+00	 1.3629054e-01	  1.2676973e+00 	 2.1678519e-01


.. parsed-literal::

      87	 1.3701798e+00	 1.3639637e-01	 1.4260760e+00	 1.3573992e-01	  1.2779773e+00 	 2.1479130e-01


.. parsed-literal::

      88	 1.3736581e+00	 1.3599468e-01	 1.4296126e+00	 1.3526355e-01	  1.2807673e+00 	 2.1139193e-01
      89	 1.3774967e+00	 1.3573105e-01	 1.4334626e+00	 1.3495264e-01	  1.2853475e+00 	 2.0989847e-01


.. parsed-literal::

      90	 1.3801949e+00	 1.3483584e-01	 1.4364688e+00	 1.3445919e-01	  1.2816746e+00 	 2.0977902e-01


.. parsed-literal::

      91	 1.3861314e+00	 1.3491854e-01	 1.4421214e+00	 1.3460134e-01	  1.2911834e+00 	 2.1496320e-01
      92	 1.3889118e+00	 1.3490902e-01	 1.4448268e+00	 1.3466657e-01	[ 1.2938496e+00]	 2.0105457e-01


.. parsed-literal::

      93	 1.3934856e+00	 1.3473266e-01	 1.4493709e+00	 1.3466892e-01	[ 1.2975318e+00]	 2.0166278e-01


.. parsed-literal::

      94	 1.3958872e+00	 1.3429688e-01	 1.4518265e+00	 1.3414107e-01	  1.2933755e+00 	 2.1161556e-01


.. parsed-literal::

      95	 1.4005157e+00	 1.3416081e-01	 1.4563154e+00	 1.3404286e-01	[ 1.2998259e+00]	 2.1518373e-01


.. parsed-literal::

      96	 1.4030443e+00	 1.3395970e-01	 1.4588170e+00	 1.3375552e-01	[ 1.3019316e+00]	 2.0967054e-01


.. parsed-literal::

      97	 1.4062713e+00	 1.3360314e-01	 1.4620659e+00	 1.3325916e-01	[ 1.3026643e+00]	 2.1403146e-01


.. parsed-literal::

      98	 1.4106418e+00	 1.3309181e-01	 1.4664887e+00	 1.3249958e-01	[ 1.3064732e+00]	 2.1594572e-01


.. parsed-literal::

      99	 1.4117438e+00	 1.3248155e-01	 1.4678740e+00	 1.3191728e-01	  1.2980394e+00 	 2.1303034e-01
     100	 1.4157221e+00	 1.3247190e-01	 1.4716629e+00	 1.3178319e-01	[ 1.3074594e+00]	 1.8828034e-01


.. parsed-literal::

     101	 1.4171905e+00	 1.3242699e-01	 1.4731154e+00	 1.3170848e-01	[ 1.3099392e+00]	 2.1229911e-01


.. parsed-literal::

     102	 1.4205861e+00	 1.3224666e-01	 1.4766044e+00	 1.3131157e-01	[ 1.3113036e+00]	 2.1306491e-01


.. parsed-literal::

     103	 1.4241907e+00	 1.3190282e-01	 1.4803979e+00	 1.3068049e-01	  1.3102145e+00 	 2.0923233e-01


.. parsed-literal::

     104	 1.4274487e+00	 1.3178410e-01	 1.4838020e+00	 1.3029882e-01	  1.3049854e+00 	 2.4695826e-01


.. parsed-literal::

     105	 1.4297112e+00	 1.3156401e-01	 1.4860191e+00	 1.3011107e-01	  1.3070532e+00 	 2.1216297e-01
     106	 1.4322701e+00	 1.3121080e-01	 1.4885978e+00	 1.2982536e-01	  1.3081880e+00 	 2.0281911e-01


.. parsed-literal::

     107	 1.4348930e+00	 1.3078348e-01	 1.4912927e+00	 1.2942925e-01	  1.3076497e+00 	 2.1841359e-01


.. parsed-literal::

     108	 1.4382206e+00	 1.3029578e-01	 1.4947848e+00	 1.2901031e-01	[ 1.3121489e+00]	 2.2426200e-01


.. parsed-literal::

     109	 1.4420716e+00	 1.2974419e-01	 1.4986109e+00	 1.2842053e-01	  1.3115765e+00 	 2.2189569e-01


.. parsed-literal::

     110	 1.4446342e+00	 1.2968977e-01	 1.5010819e+00	 1.2837450e-01	[ 1.3154605e+00]	 2.1494699e-01


.. parsed-literal::

     111	 1.4478224e+00	 1.2942403e-01	 1.5043011e+00	 1.2825060e-01	[ 1.3170245e+00]	 2.1261644e-01


.. parsed-literal::

     112	 1.4501493e+00	 1.2922135e-01	 1.5068003e+00	 1.2838269e-01	  1.3125031e+00 	 2.1873188e-01


.. parsed-literal::

     113	 1.4531039e+00	 1.2902836e-01	 1.5097333e+00	 1.2831336e-01	  1.3131282e+00 	 2.1140838e-01


.. parsed-literal::

     114	 1.4553709e+00	 1.2880230e-01	 1.5120710e+00	 1.2832290e-01	  1.3107309e+00 	 2.1144867e-01


.. parsed-literal::

     115	 1.4570186e+00	 1.2865933e-01	 1.5137633e+00	 1.2831308e-01	  1.3092120e+00 	 2.0948553e-01


.. parsed-literal::

     116	 1.4603081e+00	 1.2825333e-01	 1.5172422e+00	 1.2862613e-01	  1.3002730e+00 	 2.0863485e-01


.. parsed-literal::

     117	 1.4623778e+00	 1.2814438e-01	 1.5194070e+00	 1.2864465e-01	  1.2952247e+00 	 2.1341968e-01


.. parsed-literal::

     118	 1.4645327e+00	 1.2809911e-01	 1.5214173e+00	 1.2853184e-01	  1.3015500e+00 	 2.0700026e-01
     119	 1.4661825e+00	 1.2799470e-01	 1.5230452e+00	 1.2848659e-01	  1.3028851e+00 	 2.0723367e-01


.. parsed-literal::

     120	 1.4683871e+00	 1.2780797e-01	 1.5252744e+00	 1.2847338e-01	  1.3018336e+00 	 2.1111131e-01


.. parsed-literal::

     121	 1.4690876e+00	 1.2749499e-01	 1.5261975e+00	 1.2811039e-01	  1.3008958e+00 	 2.0898581e-01


.. parsed-literal::

     122	 1.4723120e+00	 1.2738317e-01	 1.5293117e+00	 1.2835726e-01	  1.2989687e+00 	 2.0419598e-01


.. parsed-literal::

     123	 1.4738152e+00	 1.2726754e-01	 1.5308004e+00	 1.2839039e-01	  1.2979289e+00 	 2.1344423e-01


.. parsed-literal::

     124	 1.4759052e+00	 1.2704310e-01	 1.5329034e+00	 1.2839197e-01	  1.2966399e+00 	 2.1361780e-01
     125	 1.4789397e+00	 1.2675636e-01	 1.5359284e+00	 1.2840056e-01	  1.2958009e+00 	 2.0355415e-01


.. parsed-literal::

     126	 1.4813467e+00	 1.2646085e-01	 1.5384415e+00	 1.2865631e-01	  1.2951411e+00 	 3.3041430e-01
     127	 1.4847685e+00	 1.2621670e-01	 1.5418809e+00	 1.2867486e-01	  1.2936183e+00 	 2.0061469e-01


.. parsed-literal::

     128	 1.4866766e+00	 1.2609086e-01	 1.5438355e+00	 1.2871275e-01	  1.2894410e+00 	 2.1123648e-01


.. parsed-literal::

     129	 1.4890342e+00	 1.2599011e-01	 1.5462975e+00	 1.2872509e-01	  1.2877218e+00 	 2.0532894e-01
     130	 1.4907925e+00	 1.2589811e-01	 1.5481679e+00	 1.2881627e-01	  1.2805568e+00 	 2.0586228e-01


.. parsed-literal::

     131	 1.4927110e+00	 1.2576784e-01	 1.5500898e+00	 1.2894969e-01	  1.2785140e+00 	 2.0016026e-01
     132	 1.4943290e+00	 1.2569797e-01	 1.5517100e+00	 1.2895307e-01	  1.2795046e+00 	 1.9437790e-01


.. parsed-literal::

     133	 1.4961084e+00	 1.2556125e-01	 1.5535233e+00	 1.2897178e-01	  1.2785410e+00 	 2.0144033e-01


.. parsed-literal::

     134	 1.4978326e+00	 1.2544120e-01	 1.5554566e+00	 1.2936699e-01	  1.2720244e+00 	 2.1444726e-01


.. parsed-literal::

     135	 1.5003771e+00	 1.2526966e-01	 1.5579809e+00	 1.2917379e-01	  1.2763290e+00 	 2.1030116e-01


.. parsed-literal::

     136	 1.5018040e+00	 1.2517297e-01	 1.5594301e+00	 1.2920232e-01	  1.2761416e+00 	 2.0408893e-01
     137	 1.5033795e+00	 1.2504606e-01	 1.5611057e+00	 1.2916013e-01	  1.2767876e+00 	 1.8006420e-01


.. parsed-literal::

     138	 1.5051761e+00	 1.2492547e-01	 1.5629974e+00	 1.2935382e-01	  1.2730219e+00 	 2.0867729e-01


.. parsed-literal::

     139	 1.5067914e+00	 1.2488060e-01	 1.5646025e+00	 1.2929677e-01	  1.2748955e+00 	 2.1029472e-01


.. parsed-literal::

     140	 1.5085753e+00	 1.2482268e-01	 1.5663706e+00	 1.2925473e-01	  1.2764532e+00 	 2.1980762e-01


.. parsed-literal::

     141	 1.5098700e+00	 1.2476700e-01	 1.5676264e+00	 1.2920329e-01	  1.2782274e+00 	 2.1022630e-01


.. parsed-literal::

     142	 1.5119102e+00	 1.2464344e-01	 1.5696494e+00	 1.2934927e-01	  1.2780958e+00 	 2.1620417e-01
     143	 1.5134622e+00	 1.2451609e-01	 1.5711399e+00	 1.2921330e-01	  1.2769281e+00 	 2.0836210e-01


.. parsed-literal::

     144	 1.5144500e+00	 1.2442168e-01	 1.5721393e+00	 1.2937348e-01	  1.2746388e+00 	 2.1316695e-01
     145	 1.5156811e+00	 1.2433007e-01	 1.5734307e+00	 1.2951716e-01	  1.2716087e+00 	 1.8949890e-01


.. parsed-literal::

     146	 1.5169783e+00	 1.2418678e-01	 1.5748334e+00	 1.2971738e-01	  1.2679227e+00 	 2.4506450e-01
     147	 1.5185302e+00	 1.2408356e-01	 1.5764624e+00	 1.2974074e-01	  1.2632962e+00 	 2.0195246e-01


.. parsed-literal::

     148	 1.5200932e+00	 1.2392390e-01	 1.5780870e+00	 1.2957608e-01	  1.2587473e+00 	 2.1332026e-01


.. parsed-literal::

     149	 1.5216841e+00	 1.2376125e-01	 1.5796954e+00	 1.2952673e-01	  1.2498624e+00 	 2.1356487e-01


.. parsed-literal::

     150	 1.5230702e+00	 1.2360052e-01	 1.5810895e+00	 1.2924768e-01	  1.2456605e+00 	 2.2051954e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 9s, sys: 1.25 s, total: 2min 10s
    Wall time: 32.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f9dd40b0040>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 986 ms, sys: 41 ms, total: 1.03 s
    Wall time: 401 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

