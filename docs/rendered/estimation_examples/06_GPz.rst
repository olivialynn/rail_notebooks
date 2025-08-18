GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

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
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

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
       1	-3.3150776e-01	 3.1769633e-01	-3.2173191e-01	 3.3237115e-01	[-3.5032745e-01]	 4.6452427e-01


.. parsed-literal::

       2	-2.6358847e-01	 3.0706465e-01	-2.3913136e-01	 3.2029612e-01	[-2.8124472e-01]	 2.3104692e-01


.. parsed-literal::

       3	-2.2202965e-01	 2.8755648e-01	-1.7992661e-01	 3.0150786e-01	[-2.3427472e-01]	 2.9215503e-01
       4	-1.8259408e-01	 2.6467735e-01	-1.4089612e-01	 2.7191762e-01	[-1.8248198e-01]	 1.8322921e-01


.. parsed-literal::

       5	-9.5166838e-02	 2.5423163e-01	-6.1204686e-02	 2.6300747e-01	[-1.0050243e-01]	 2.1661520e-01
       6	-6.3425008e-02	 2.5030365e-01	-3.3817495e-02	 2.5832810e-01	[-6.5135284e-02]	 1.8101835e-01


.. parsed-literal::

       7	-4.5881923e-02	 2.4722067e-01	-2.1698406e-02	 2.5459085e-01	[-5.1469600e-02]	 2.0969987e-01


.. parsed-literal::

       8	-3.5162494e-02	 2.4559288e-01	-1.4777540e-02	 2.5264609e-01	[-4.3826632e-02]	 2.1112251e-01


.. parsed-literal::

       9	-2.3937079e-02	 2.4359552e-01	-6.2450281e-03	 2.5074036e-01	[-3.4323236e-02]	 2.1120787e-01


.. parsed-literal::

      10	-1.5658723e-02	 2.4151973e-01	-3.2505505e-04	 2.4844296e-01	[-3.2461995e-02]	 2.1251297e-01


.. parsed-literal::

      11	-6.1339569e-03	 2.4034549e-01	 8.3617725e-03	 2.4719079e-01	[-1.9297319e-02]	 2.2054124e-01


.. parsed-literal::

      12	-3.5308436e-03	 2.3989307e-01	 1.0674405e-02	 2.4689993e-01	[-1.7700042e-02]	 2.1493697e-01


.. parsed-literal::

      13	 2.2758852e-03	 2.3883927e-01	 1.6380811e-02	 2.4561998e-01	[-1.0886241e-02]	 2.0911264e-01


.. parsed-literal::

      14	 9.9316121e-02	 2.2591925e-01	 1.1910375e-01	 2.3347212e-01	[ 9.2176884e-02]	 3.2817650e-01


.. parsed-literal::

      15	 1.1781374e-01	 2.2356650e-01	 1.4115731e-01	 2.2945750e-01	[ 1.2912268e-01]	 2.1141863e-01


.. parsed-literal::

      16	 2.2263668e-01	 2.1733614e-01	 2.4956157e-01	 2.2360762e-01	[ 2.3519272e-01]	 2.1304560e-01


.. parsed-literal::

      17	 2.9640088e-01	 2.1266460e-01	 3.2684811e-01	 2.1778209e-01	[ 3.1193956e-01]	 3.2063198e-01


.. parsed-literal::

      18	 3.4519295e-01	 2.0793449e-01	 3.7837721e-01	 2.1387313e-01	[ 3.6301382e-01]	 2.1494913e-01


.. parsed-literal::

      19	 3.8913891e-01	 2.0553377e-01	 4.2343161e-01	 2.1164617e-01	[ 4.0994380e-01]	 2.1478653e-01


.. parsed-literal::

      20	 4.2304512e-01	 2.0431943e-01	 4.5772611e-01	 2.0936580e-01	[ 4.4663801e-01]	 2.1662235e-01


.. parsed-literal::

      21	 4.5503362e-01	 2.0233684e-01	 4.9009153e-01	 2.0768214e-01	[ 4.7936713e-01]	 2.1013427e-01


.. parsed-literal::

      22	 5.7300570e-01	 1.9746420e-01	 6.1013341e-01	 2.0351553e-01	[ 6.0183538e-01]	 2.2165918e-01
      23	 5.9720419e-01	 1.9682813e-01	 6.3995438e-01	 2.0270089e-01	[ 6.4564851e-01]	 1.8547511e-01


.. parsed-literal::

      24	 6.6740316e-01	 1.8911747e-01	 7.0549601e-01	 1.9634403e-01	[ 6.9264094e-01]	 2.1607018e-01


.. parsed-literal::

      25	 6.9590520e-01	 1.8705889e-01	 7.3404562e-01	 1.9430087e-01	[ 7.1801965e-01]	 2.1221685e-01


.. parsed-literal::

      26	 7.2093366e-01	 1.8592571e-01	 7.5890540e-01	 1.9297344e-01	[ 7.4097193e-01]	 2.1116614e-01
      27	 7.4838967e-01	 1.8332583e-01	 7.8677975e-01	 1.9093048e-01	[ 7.6699189e-01]	 1.8326211e-01


.. parsed-literal::

      28	 7.7833471e-01	 1.8433484e-01	 8.1703703e-01	 1.9340245e-01	[ 7.9094051e-01]	 2.0793056e-01


.. parsed-literal::

      29	 8.1318546e-01	 1.7992012e-01	 8.5355405e-01	 1.8949875e-01	[ 8.3276908e-01]	 2.0676637e-01


.. parsed-literal::

      30	 8.4891664e-01	 1.8231053e-01	 8.8855558e-01	 1.9225199e-01	[ 8.5491544e-01]	 2.0581627e-01
      31	 8.7008779e-01	 1.8488651e-01	 9.0954608e-01	 1.9463762e-01	[ 8.7083379e-01]	 1.7711329e-01


.. parsed-literal::

      32	 8.9814472e-01	 1.8051691e-01	 9.3884253e-01	 1.8982005e-01	[ 9.0160969e-01]	 2.0577192e-01


.. parsed-literal::

      33	 9.1938062e-01	 1.7228214e-01	 9.6098170e-01	 1.8032642e-01	[ 9.3485157e-01]	 2.1345043e-01


.. parsed-literal::

      34	 9.4564143e-01	 1.6819720e-01	 9.8727370e-01	 1.7719088e-01	[ 9.6129433e-01]	 2.1957207e-01


.. parsed-literal::

      35	 9.5913665e-01	 1.6589473e-01	 1.0008260e+00	 1.7568992e-01	[ 9.7300337e-01]	 2.0705080e-01


.. parsed-literal::

      36	 9.7560803e-01	 1.6339712e-01	 1.0178468e+00	 1.7346481e-01	[ 9.8761615e-01]	 2.1313977e-01
      37	 1.0002773e+00	 1.6002136e-01	 1.0436725e+00	 1.7040557e-01	[ 1.0075276e+00]	 1.7766738e-01


.. parsed-literal::

      38	 1.0098271e+00	 1.6154528e-01	 1.0553600e+00	 1.6903520e-01	[ 1.0098581e+00]	 2.1826100e-01
      39	 1.0291690e+00	 1.5910092e-01	 1.0742181e+00	 1.6720197e-01	[ 1.0266379e+00]	 1.7610192e-01


.. parsed-literal::

      40	 1.0420913e+00	 1.5704332e-01	 1.0871504e+00	 1.6628981e-01	[ 1.0421715e+00]	 1.7754984e-01
      41	 1.0547478e+00	 1.5579991e-01	 1.0999968e+00	 1.6621021e-01	[ 1.0524701e+00]	 1.7596650e-01


.. parsed-literal::

      42	 1.0757968e+00	 1.5456214e-01	 1.1220022e+00	 1.6638747e-01	[ 1.0674639e+00]	 2.0132279e-01


.. parsed-literal::

      43	 1.0910564e+00	 1.5240900e-01	 1.1375919e+00	 1.6673500e-01	[ 1.0744019e+00]	 2.1204376e-01
      44	 1.1021565e+00	 1.4990028e-01	 1.1489062e+00	 1.6482613e-01	[ 1.0869854e+00]	 1.6520381e-01


.. parsed-literal::

      45	 1.1157102e+00	 1.4590830e-01	 1.1625619e+00	 1.6093825e-01	[ 1.1024529e+00]	 2.0882130e-01


.. parsed-literal::

      46	 1.1295712e+00	 1.4265378e-01	 1.1766793e+00	 1.5860877e-01	[ 1.1158053e+00]	 2.1362424e-01


.. parsed-literal::

      47	 1.1405658e+00	 1.3977787e-01	 1.1875186e+00	 1.5573949e-01	[ 1.1223428e+00]	 2.1294093e-01


.. parsed-literal::

      48	 1.1506912e+00	 1.3939690e-01	 1.1972850e+00	 1.5551103e-01	[ 1.1322423e+00]	 2.1432829e-01


.. parsed-literal::

      49	 1.1586997e+00	 1.3876651e-01	 1.2053925e+00	 1.5595587e-01	[ 1.1347451e+00]	 2.0643330e-01


.. parsed-literal::

      50	 1.1688767e+00	 1.3777939e-01	 1.2158689e+00	 1.5532488e-01	[ 1.1387861e+00]	 2.0561671e-01
      51	 1.1803973e+00	 1.3739507e-01	 1.2277246e+00	 1.5470246e-01	[ 1.1486560e+00]	 1.9523692e-01


.. parsed-literal::

      52	 1.1921184e+00	 1.3625833e-01	 1.2396623e+00	 1.5428334e-01	[ 1.1532849e+00]	 1.7757273e-01
      53	 1.2041564e+00	 1.3560626e-01	 1.2520413e+00	 1.5320884e-01	[ 1.1627469e+00]	 1.9864178e-01


.. parsed-literal::

      54	 1.2110771e+00	 1.3439087e-01	 1.2594508e+00	 1.5358229e-01	  1.1602401e+00 	 2.0152259e-01
      55	 1.2197564e+00	 1.3424060e-01	 1.2677218e+00	 1.5288679e-01	[ 1.1729327e+00]	 1.8520689e-01


.. parsed-literal::

      56	 1.2257724e+00	 1.3415886e-01	 1.2738747e+00	 1.5253016e-01	[ 1.1782244e+00]	 2.0606422e-01


.. parsed-literal::

      57	 1.2375557e+00	 1.3383298e-01	 1.2861096e+00	 1.5192474e-01	[ 1.1845698e+00]	 2.0901155e-01
      58	 1.2464474e+00	 1.3380135e-01	 1.2948788e+00	 1.5113588e-01	[ 1.1919561e+00]	 1.9949245e-01


.. parsed-literal::

      59	 1.2554374e+00	 1.3322021e-01	 1.3038885e+00	 1.5051476e-01	[ 1.1954222e+00]	 2.1121383e-01


.. parsed-literal::

      60	 1.2623774e+00	 1.3299779e-01	 1.3108815e+00	 1.4989732e-01	[ 1.2030727e+00]	 2.2134995e-01


.. parsed-literal::

      61	 1.2725676e+00	 1.3231229e-01	 1.3215497e+00	 1.4908515e-01	[ 1.2035035e+00]	 2.1741557e-01


.. parsed-literal::

      62	 1.2811977e+00	 1.3228823e-01	 1.3303893e+00	 1.4849519e-01	[ 1.2065588e+00]	 2.0870590e-01
      63	 1.2876550e+00	 1.3199669e-01	 1.3368594e+00	 1.4837191e-01	[ 1.2109029e+00]	 1.8957305e-01


.. parsed-literal::

      64	 1.2959912e+00	 1.3171867e-01	 1.3454192e+00	 1.4828410e-01	[ 1.2133973e+00]	 2.1108484e-01


.. parsed-literal::

      65	 1.3049086e+00	 1.3110481e-01	 1.3545517e+00	 1.4777242e-01	[ 1.2181547e+00]	 2.0224595e-01


.. parsed-literal::

      66	 1.3141467e+00	 1.3073530e-01	 1.3640175e+00	 1.4715096e-01	[ 1.2274641e+00]	 2.2356820e-01


.. parsed-literal::

      67	 1.3216338e+00	 1.3044934e-01	 1.3717152e+00	 1.4639585e-01	[ 1.2352006e+00]	 2.0356965e-01


.. parsed-literal::

      68	 1.3282206e+00	 1.3024686e-01	 1.3784684e+00	 1.4602714e-01	[ 1.2446093e+00]	 2.2205925e-01


.. parsed-literal::

      69	 1.3342323e+00	 1.3004618e-01	 1.3844843e+00	 1.4616692e-01	[ 1.2503383e+00]	 2.0914745e-01
      70	 1.3407502e+00	 1.2986478e-01	 1.3910553e+00	 1.4621797e-01	[ 1.2534095e+00]	 1.9745922e-01


.. parsed-literal::

      71	 1.3483832e+00	 1.2955091e-01	 1.3990674e+00	 1.4643117e-01	  1.2504904e+00 	 1.6900182e-01


.. parsed-literal::

      72	 1.3557143e+00	 1.2931154e-01	 1.4063804e+00	 1.4612274e-01	  1.2520176e+00 	 2.1398759e-01
      73	 1.3590552e+00	 1.2901317e-01	 1.4096423e+00	 1.4563564e-01	[ 1.2542716e+00]	 1.8927526e-01


.. parsed-literal::

      74	 1.3663215e+00	 1.2840893e-01	 1.4170641e+00	 1.4425834e-01	[ 1.2564822e+00]	 2.1896315e-01


.. parsed-literal::

      75	 1.3696872e+00	 1.2809417e-01	 1.4205937e+00	 1.4369688e-01	[ 1.2589390e+00]	 2.0973468e-01


.. parsed-literal::

      76	 1.3742425e+00	 1.2804029e-01	 1.4250676e+00	 1.4353655e-01	[ 1.2640567e+00]	 2.1315932e-01


.. parsed-literal::

      77	 1.3781261e+00	 1.2795396e-01	 1.4289632e+00	 1.4313690e-01	[ 1.2694451e+00]	 2.0511413e-01


.. parsed-literal::

      78	 1.3808732e+00	 1.2784554e-01	 1.4317433e+00	 1.4276364e-01	[ 1.2730022e+00]	 2.2090244e-01


.. parsed-literal::

      79	 1.3878305e+00	 1.2757067e-01	 1.4389664e+00	 1.4198905e-01	[ 1.2788406e+00]	 2.1757388e-01


.. parsed-literal::

      80	 1.3919946e+00	 1.2736567e-01	 1.4433121e+00	 1.4124851e-01	[ 1.2809958e+00]	 3.0220819e-01


.. parsed-literal::

      81	 1.3964948e+00	 1.2718224e-01	 1.4479247e+00	 1.4111736e-01	  1.2802184e+00 	 2.1321297e-01


.. parsed-literal::

      82	 1.4000393e+00	 1.2705089e-01	 1.4515738e+00	 1.4131195e-01	  1.2760807e+00 	 2.1599436e-01
      83	 1.4033517e+00	 1.2698523e-01	 1.4550183e+00	 1.4175729e-01	  1.2718208e+00 	 1.9544625e-01


.. parsed-literal::

      84	 1.4073464e+00	 1.2701794e-01	 1.4590566e+00	 1.4212347e-01	  1.2711819e+00 	 2.1733975e-01


.. parsed-literal::

      85	 1.4131687e+00	 1.2704654e-01	 1.4650895e+00	 1.4252959e-01	  1.2682791e+00 	 2.2088313e-01


.. parsed-literal::

      86	 1.4151231e+00	 1.2753866e-01	 1.4671520e+00	 1.4329511e-01	  1.2687393e+00 	 2.1091175e-01


.. parsed-literal::

      87	 1.4187061e+00	 1.2715669e-01	 1.4706001e+00	 1.4262185e-01	  1.2739034e+00 	 2.1286225e-01
      88	 1.4212956e+00	 1.2685060e-01	 1.4731471e+00	 1.4204061e-01	  1.2763723e+00 	 1.8002152e-01


.. parsed-literal::

      89	 1.4244354e+00	 1.2651445e-01	 1.4762534e+00	 1.4157132e-01	  1.2778576e+00 	 1.8494320e-01


.. parsed-literal::

      90	 1.4291543e+00	 1.2610982e-01	 1.4810125e+00	 1.4104934e-01	  1.2774039e+00 	 2.1711779e-01
      91	 1.4309026e+00	 1.2571495e-01	 1.4829201e+00	 1.4155433e-01	  1.2694628e+00 	 2.0838833e-01


.. parsed-literal::

      92	 1.4351148e+00	 1.2567003e-01	 1.4869692e+00	 1.4129776e-01	  1.2770703e+00 	 2.0586443e-01
      93	 1.4371233e+00	 1.2564596e-01	 1.4889831e+00	 1.4138014e-01	  1.2790713e+00 	 1.7388296e-01


.. parsed-literal::

      94	 1.4393983e+00	 1.2551242e-01	 1.4912929e+00	 1.4140533e-01	  1.2803191e+00 	 1.7606759e-01


.. parsed-literal::

      95	 1.4433932e+00	 1.2522783e-01	 1.4954399e+00	 1.4142873e-01	[ 1.2812009e+00]	 2.2086596e-01


.. parsed-literal::

      96	 1.4457162e+00	 1.2494190e-01	 1.4978504e+00	 1.4113674e-01	  1.2808109e+00 	 3.2640553e-01


.. parsed-literal::

      97	 1.4480106e+00	 1.2477537e-01	 1.5001571e+00	 1.4098230e-01	[ 1.2824418e+00]	 2.1882653e-01


.. parsed-literal::

      98	 1.4502388e+00	 1.2455424e-01	 1.5024038e+00	 1.4071360e-01	[ 1.2840649e+00]	 2.1649909e-01
      99	 1.4527355e+00	 1.2426304e-01	 1.5049315e+00	 1.4047540e-01	[ 1.2860598e+00]	 1.7256093e-01


.. parsed-literal::

     100	 1.4562923e+00	 1.2370006e-01	 1.5086885e+00	 1.4017728e-01	  1.2858097e+00 	 2.1613050e-01


.. parsed-literal::

     101	 1.4597013e+00	 1.2334392e-01	 1.5120797e+00	 1.3981681e-01	[ 1.2914076e+00]	 2.1562076e-01
     102	 1.4613195e+00	 1.2332458e-01	 1.5136629e+00	 1.3974880e-01	[ 1.2922529e+00]	 1.8627691e-01


.. parsed-literal::

     103	 1.4641374e+00	 1.2327007e-01	 1.5165316e+00	 1.3958365e-01	[ 1.2940173e+00]	 1.7761850e-01
     104	 1.4662349e+00	 1.2318046e-01	 1.5187006e+00	 1.3950372e-01	  1.2920733e+00 	 1.8840241e-01


.. parsed-literal::

     105	 1.4691274e+00	 1.2312326e-01	 1.5217410e+00	 1.3933804e-01	  1.2924613e+00 	 2.1261883e-01


.. parsed-literal::

     106	 1.4706103e+00	 1.2305855e-01	 1.5233071e+00	 1.3932587e-01	  1.2896324e+00 	 2.2065067e-01


.. parsed-literal::

     107	 1.4729850e+00	 1.2302013e-01	 1.5255818e+00	 1.3925592e-01	  1.2919764e+00 	 2.1723819e-01


.. parsed-literal::

     108	 1.4744371e+00	 1.2299783e-01	 1.5269766e+00	 1.3920672e-01	  1.2914305e+00 	 2.0502496e-01
     109	 1.4767318e+00	 1.2296551e-01	 1.5292119e+00	 1.3911172e-01	  1.2898184e+00 	 1.8991780e-01


.. parsed-literal::

     110	 1.4786227e+00	 1.2304096e-01	 1.5310969e+00	 1.3889270e-01	  1.2812327e+00 	 2.1706343e-01
     111	 1.4804780e+00	 1.2298985e-01	 1.5329271e+00	 1.3878578e-01	  1.2828544e+00 	 1.9999766e-01


.. parsed-literal::

     112	 1.4820675e+00	 1.2295434e-01	 1.5345548e+00	 1.3860962e-01	  1.2822157e+00 	 2.2094536e-01


.. parsed-literal::

     113	 1.4836584e+00	 1.2289786e-01	 1.5361858e+00	 1.3841228e-01	  1.2806827e+00 	 2.1975017e-01


.. parsed-literal::

     114	 1.4872062e+00	 1.2265655e-01	 1.5398655e+00	 1.3815293e-01	  1.2740055e+00 	 2.2368383e-01


.. parsed-literal::

     115	 1.4888859e+00	 1.2253321e-01	 1.5416109e+00	 1.3779524e-01	  1.2739411e+00 	 3.2640195e-01


.. parsed-literal::

     116	 1.4909837e+00	 1.2234435e-01	 1.5437481e+00	 1.3768104e-01	  1.2707943e+00 	 2.1399665e-01


.. parsed-literal::

     117	 1.4932143e+00	 1.2210101e-01	 1.5460007e+00	 1.3757265e-01	  1.2683891e+00 	 2.1323776e-01
     118	 1.4952089e+00	 1.2191751e-01	 1.5479980e+00	 1.3736666e-01	  1.2647360e+00 	 2.0582962e-01


.. parsed-literal::

     119	 1.4970028e+00	 1.2176793e-01	 1.5498940e+00	 1.3708229e-01	  1.2590572e+00 	 1.7690277e-01
     120	 1.4991760e+00	 1.2167674e-01	 1.5520232e+00	 1.3680253e-01	  1.2579047e+00 	 1.8187642e-01


.. parsed-literal::

     121	 1.5006526e+00	 1.2169814e-01	 1.5535198e+00	 1.3656578e-01	  1.2558997e+00 	 2.2642756e-01


.. parsed-literal::

     122	 1.5019624e+00	 1.2173754e-01	 1.5548950e+00	 1.3628398e-01	  1.2518268e+00 	 2.0937848e-01


.. parsed-literal::

     123	 1.5033165e+00	 1.2179179e-01	 1.5563277e+00	 1.3618978e-01	  1.2481403e+00 	 2.0563030e-01


.. parsed-literal::

     124	 1.5044106e+00	 1.2175476e-01	 1.5574451e+00	 1.3615497e-01	  1.2482499e+00 	 2.1067286e-01
     125	 1.5062075e+00	 1.2163868e-01	 1.5593388e+00	 1.3601573e-01	  1.2462891e+00 	 2.0472908e-01


.. parsed-literal::

     126	 1.5066974e+00	 1.2160438e-01	 1.5599313e+00	 1.3584875e-01	  1.2456352e+00 	 2.1159220e-01


.. parsed-literal::

     127	 1.5082034e+00	 1.2155384e-01	 1.5613536e+00	 1.3578564e-01	  1.2468176e+00 	 2.0897174e-01


.. parsed-literal::

     128	 1.5094049e+00	 1.2149936e-01	 1.5625382e+00	 1.3561348e-01	  1.2467760e+00 	 2.0254207e-01
     129	 1.5106205e+00	 1.2142209e-01	 1.5637640e+00	 1.3536081e-01	  1.2465670e+00 	 1.7476845e-01


.. parsed-literal::

     130	 1.5122377e+00	 1.2127605e-01	 1.5653820e+00	 1.3510135e-01	  1.2478996e+00 	 2.1104574e-01
     131	 1.5141692e+00	 1.2099238e-01	 1.5673787e+00	 1.3478513e-01	  1.2468022e+00 	 1.7034602e-01


.. parsed-literal::

     132	 1.5159124e+00	 1.2082436e-01	 1.5691177e+00	 1.3463814e-01	  1.2509621e+00 	 1.9157577e-01
     133	 1.5169754e+00	 1.2083677e-01	 1.5701306e+00	 1.3479574e-01	  1.2512798e+00 	 1.8117213e-01


.. parsed-literal::

     134	 1.5181259e+00	 1.2082437e-01	 1.5712830e+00	 1.3493571e-01	  1.2507341e+00 	 1.7555857e-01
     135	 1.5197388e+00	 1.2081098e-01	 1.5729459e+00	 1.3506249e-01	  1.2476649e+00 	 1.7173576e-01


.. parsed-literal::

     136	 1.5204643e+00	 1.2094082e-01	 1.5738403e+00	 1.3527028e-01	  1.2423100e+00 	 2.0199084e-01


.. parsed-literal::

     137	 1.5225580e+00	 1.2087008e-01	 1.5758907e+00	 1.3515896e-01	  1.2413458e+00 	 2.0838380e-01


.. parsed-literal::

     138	 1.5235818e+00	 1.2083216e-01	 1.5769078e+00	 1.3500683e-01	  1.2412742e+00 	 2.1888113e-01


.. parsed-literal::

     139	 1.5248289e+00	 1.2084242e-01	 1.5781870e+00	 1.3492600e-01	  1.2395608e+00 	 2.1470118e-01
     140	 1.5263278e+00	 1.2082745e-01	 1.5797566e+00	 1.3490385e-01	  1.2357955e+00 	 1.8196225e-01


.. parsed-literal::

     141	 1.5279351e+00	 1.2084112e-01	 1.5813578e+00	 1.3501615e-01	  1.2353917e+00 	 2.1126914e-01
     142	 1.5294600e+00	 1.2083084e-01	 1.5829046e+00	 1.3524825e-01	  1.2332151e+00 	 1.9289994e-01


.. parsed-literal::

     143	 1.5303805e+00	 1.2073403e-01	 1.5838584e+00	 1.3532677e-01	  1.2344128e+00 	 2.0307088e-01


.. parsed-literal::

     144	 1.5314945e+00	 1.2065064e-01	 1.5849904e+00	 1.3535474e-01	  1.2333867e+00 	 2.0831561e-01


.. parsed-literal::

     145	 1.5329386e+00	 1.2050374e-01	 1.5864872e+00	 1.3534558e-01	  1.2321994e+00 	 2.1614885e-01


.. parsed-literal::

     146	 1.5341549e+00	 1.2031682e-01	 1.5877501e+00	 1.3527655e-01	  1.2289095e+00 	 2.1088600e-01


.. parsed-literal::

     147	 1.5357304e+00	 1.2011666e-01	 1.5893629e+00	 1.3528217e-01	  1.2256188e+00 	 2.1846247e-01


.. parsed-literal::

     148	 1.5372056e+00	 1.1993830e-01	 1.5908490e+00	 1.3533681e-01	  1.2199525e+00 	 2.0522428e-01
     149	 1.5389150e+00	 1.1971954e-01	 1.5926031e+00	 1.3548834e-01	  1.2109974e+00 	 1.7556405e-01


.. parsed-literal::

     150	 1.5393419e+00	 1.1960184e-01	 1.5931687e+00	 1.3582153e-01	  1.1915415e+00 	 2.0892668e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.17 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd9b0c47790>



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

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.04 s, sys: 50 ms, total: 2.09 s
    Wall time: 620 ms


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




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

