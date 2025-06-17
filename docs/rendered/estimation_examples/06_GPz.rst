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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.5289954e-01	 3.2341524e-01	-3.4323796e-01	 3.0858278e-01	[-3.1625273e-01]	 4.7612834e-01


.. parsed-literal::

       2	-2.8275977e-01	 3.1328907e-01	-2.5941580e-01	 2.9948166e-01	[-2.1883776e-01]	 2.3207521e-01


.. parsed-literal::

       3	-2.3836193e-01	 2.9216687e-01	-1.9677112e-01	 2.7687431e-01	[-1.3757471e-01]	 2.9117632e-01
       4	-1.9904846e-01	 2.6873654e-01	-1.5794731e-01	 2.5517467e-01	[-8.5033981e-02]	 1.9480276e-01


.. parsed-literal::

       5	-1.1013061e-01	 2.5895661e-01	-7.7528358e-02	 2.4733306e-01	[-2.5504201e-02]	 2.0789623e-01
       6	-7.9935576e-02	 2.5447228e-01	-5.0988084e-02	 2.4184709e-01	[-6.3503761e-03]	 1.9135761e-01


.. parsed-literal::

       7	-6.2002670e-02	 2.5145840e-01	-3.8747497e-02	 2.3882322e-01	[ 9.2211143e-03]	 1.9612837e-01


.. parsed-literal::

       8	-4.9623797e-02	 2.4933763e-01	-2.9985537e-02	 2.3613186e-01	[ 2.1384019e-02]	 2.0292783e-01


.. parsed-literal::

       9	-3.7387439e-02	 2.4703455e-01	-2.0240727e-02	 2.3366762e-01	[ 3.2525007e-02]	 2.1739554e-01
      10	-2.4977548e-02	 2.4456018e-01	-9.5570853e-03	 2.3210023e-01	[ 4.0175127e-02]	 1.7894006e-01


.. parsed-literal::

      11	-2.1874954e-02	 2.4436114e-01	-7.8448212e-03	 2.3214131e-01	[ 4.1708883e-02]	 2.2016072e-01


.. parsed-literal::

      12	-1.7949598e-02	 2.4342033e-01	-4.1296199e-03	 2.3174502e-01	  4.1301838e-02 	 2.1151638e-01


.. parsed-literal::

      13	-1.5288100e-02	 2.4290768e-01	-1.4689682e-03	 2.3151948e-01	[ 4.4881407e-02]	 2.0980620e-01
      14	-1.1827564e-02	 2.4221074e-01	 2.2196823e-03	 2.3107370e-01	[ 4.6898089e-02]	 1.9998503e-01


.. parsed-literal::

      15	 8.1318313e-02	 2.2695215e-01	 1.0013569e-01	 2.1627080e-01	[ 1.2990213e-01]	 3.3913398e-01


.. parsed-literal::

      16	 1.4056113e-01	 2.2209838e-01	 1.6194872e-01	 2.1215778e-01	[ 1.9284233e-01]	 2.1053481e-01
      17	 2.6030118e-01	 2.1959048e-01	 2.8858042e-01	 2.1079366e-01	[ 3.3045765e-01]	 1.9310689e-01


.. parsed-literal::

      18	 3.0495497e-01	 2.1651482e-01	 3.3438780e-01	 2.0880355e-01	[ 3.7495092e-01]	 2.0944405e-01


.. parsed-literal::

      19	 3.5242243e-01	 2.1275858e-01	 3.8271299e-01	 2.0698902e-01	[ 4.1281688e-01]	 2.1490598e-01


.. parsed-literal::

      20	 4.1041663e-01	 2.0699038e-01	 4.4179534e-01	 2.0315160e-01	[ 4.6591150e-01]	 2.1943426e-01
      21	 4.9046235e-01	 2.0369641e-01	 5.2493147e-01	 1.9906385e-01	[ 5.5197660e-01]	 1.8551087e-01


.. parsed-literal::

      22	 5.7143938e-01	 2.0032023e-01	 6.0812586e-01	 1.9650981e-01	[ 6.2936699e-01]	 2.1344519e-01


.. parsed-literal::

      23	 6.1700409e-01	 1.9897608e-01	 6.5482033e-01	 1.9375737e-01	[ 6.8105576e-01]	 2.0959997e-01
      24	 6.5508870e-01	 1.9488226e-01	 6.9315600e-01	 1.9001751e-01	[ 7.1708481e-01]	 1.9792533e-01


.. parsed-literal::

      25	 6.9951324e-01	 1.9297919e-01	 7.3527128e-01	 1.8688998e-01	[ 7.5645394e-01]	 2.1018171e-01


.. parsed-literal::

      26	 7.2750475e-01	 1.9587793e-01	 7.6391810e-01	 1.8862051e-01	[ 7.9196793e-01]	 3.2595563e-01
      27	 7.5124930e-01	 1.9844298e-01	 7.8782017e-01	 1.9204642e-01	[ 8.1421785e-01]	 1.8262053e-01


.. parsed-literal::

      28	 7.7884111e-01	 1.9353906e-01	 8.1623009e-01	 1.8818122e-01	[ 8.3188259e-01]	 2.0789433e-01


.. parsed-literal::

      29	 8.0908508e-01	 1.9017650e-01	 8.4629236e-01	 1.8624169e-01	[ 8.5550904e-01]	 2.2245765e-01
      30	 8.4109516e-01	 1.8702570e-01	 8.7914058e-01	 1.8627505e-01	[ 8.8341849e-01]	 1.9850373e-01


.. parsed-literal::

      31	 8.5746762e-01	 1.8490801e-01	 8.9712997e-01	 1.8355609e-01	[ 8.9124616e-01]	 2.0392823e-01
      32	 8.7117199e-01	 1.8279088e-01	 9.1048541e-01	 1.8150048e-01	[ 9.0504809e-01]	 2.0245790e-01


.. parsed-literal::

      33	 8.9247664e-01	 1.8247926e-01	 9.3199685e-01	 1.8098923e-01	[ 9.2691219e-01]	 2.0800018e-01
      34	 9.1249260e-01	 1.8165529e-01	 9.5286940e-01	 1.7844423e-01	[ 9.3733312e-01]	 1.9839120e-01


.. parsed-literal::

      35	 9.2814811e-01	 1.8334086e-01	 9.6913910e-01	 1.7947982e-01	[ 9.5309496e-01]	 2.1172047e-01


.. parsed-literal::

      36	 9.3884994e-01	 1.8260153e-01	 9.7997459e-01	 1.7874805e-01	[ 9.6261868e-01]	 2.1625423e-01


.. parsed-literal::

      37	 9.5359003e-01	 1.8138666e-01	 9.9535718e-01	 1.7715434e-01	[ 9.7208308e-01]	 2.0986581e-01


.. parsed-literal::

      38	 9.6925787e-01	 1.8102251e-01	 1.0121408e+00	 1.7574009e-01	[ 9.8859552e-01]	 2.1494079e-01


.. parsed-literal::

      39	 9.8483149e-01	 1.7975430e-01	 1.0281676e+00	 1.7430116e-01	[ 1.0077049e+00]	 2.2297096e-01


.. parsed-literal::

      40	 9.9533773e-01	 1.7813705e-01	 1.0390721e+00	 1.7235594e-01	[ 1.0200927e+00]	 2.0940447e-01


.. parsed-literal::

      41	 1.0073742e+00	 1.7723069e-01	 1.0521983e+00	 1.7039402e-01	[ 1.0337134e+00]	 2.1676278e-01


.. parsed-literal::

      42	 1.0165057e+00	 1.7524449e-01	 1.0616955e+00	 1.6905783e-01	[ 1.0407537e+00]	 2.0847845e-01
      43	 1.0218178e+00	 1.7498743e-01	 1.0669119e+00	 1.6870585e-01	[ 1.0456654e+00]	 2.0413637e-01


.. parsed-literal::

      44	 1.0327398e+00	 1.7467121e-01	 1.0776617e+00	 1.6827689e-01	[ 1.0584715e+00]	 2.1173859e-01


.. parsed-literal::

      45	 1.0445050e+00	 1.7290328e-01	 1.0894388e+00	 1.6627472e-01	[ 1.0714301e+00]	 2.2047925e-01


.. parsed-literal::

      46	 1.0547422e+00	 1.7005263e-01	 1.1002339e+00	 1.6310540e-01	[ 1.0902757e+00]	 2.1705508e-01


.. parsed-literal::

      47	 1.0629647e+00	 1.6862861e-01	 1.1083525e+00	 1.6184898e-01	[ 1.0972033e+00]	 2.1214628e-01


.. parsed-literal::

      48	 1.0693277e+00	 1.6721087e-01	 1.1149301e+00	 1.6043818e-01	[ 1.1014633e+00]	 2.1969700e-01


.. parsed-literal::

      49	 1.0787559e+00	 1.6438875e-01	 1.1248918e+00	 1.5755078e-01	[ 1.1115865e+00]	 2.1261120e-01


.. parsed-literal::

      50	 1.0869927e+00	 1.6239902e-01	 1.1336121e+00	 1.5533825e-01	[ 1.1145839e+00]	 2.1859789e-01
      51	 1.0944416e+00	 1.6158375e-01	 1.1407718e+00	 1.5480271e-01	[ 1.1190208e+00]	 1.9417381e-01


.. parsed-literal::

      52	 1.1041712e+00	 1.6063950e-01	 1.1503936e+00	 1.5400818e-01	[ 1.1282684e+00]	 1.9348025e-01
      53	 1.1099456e+00	 1.5889053e-01	 1.1564611e+00	 1.5277712e-01	[ 1.1308217e+00]	 1.9802499e-01


.. parsed-literal::

      54	 1.1172027e+00	 1.5788601e-01	 1.1637863e+00	 1.5214158e-01	[ 1.1392169e+00]	 2.0212483e-01


.. parsed-literal::

      55	 1.1237038e+00	 1.5632594e-01	 1.1706540e+00	 1.5070332e-01	[ 1.1453570e+00]	 2.1806574e-01


.. parsed-literal::

      56	 1.1305753e+00	 1.5530500e-01	 1.1777521e+00	 1.5033380e-01	[ 1.1477605e+00]	 2.0886254e-01


.. parsed-literal::

      57	 1.1388511e+00	 1.5310233e-01	 1.1866711e+00	 1.4907285e-01	[ 1.1489091e+00]	 2.0529032e-01


.. parsed-literal::

      58	 1.1458330e+00	 1.5204727e-01	 1.1938032e+00	 1.4838176e-01	[ 1.1511557e+00]	 2.0859671e-01


.. parsed-literal::

      59	 1.1510742e+00	 1.5188696e-01	 1.1990436e+00	 1.4838074e-01	[ 1.1523689e+00]	 2.0087075e-01


.. parsed-literal::

      60	 1.1584563e+00	 1.5118157e-01	 1.2067956e+00	 1.4797947e-01	[ 1.1523842e+00]	 2.1776128e-01


.. parsed-literal::

      61	 1.1632100e+00	 1.5031913e-01	 1.2116351e+00	 1.4689327e-01	[ 1.1595352e+00]	 2.0485067e-01
      62	 1.1697624e+00	 1.4964340e-01	 1.2182344e+00	 1.4650071e-01	[ 1.1661692e+00]	 1.8681598e-01


.. parsed-literal::

      63	 1.1764907e+00	 1.4829498e-01	 1.2251949e+00	 1.4545784e-01	[ 1.1735372e+00]	 2.0728469e-01


.. parsed-literal::

      64	 1.1814923e+00	 1.4782976e-01	 1.2302610e+00	 1.4501066e-01	[ 1.1777879e+00]	 2.1400905e-01
      65	 1.1901730e+00	 1.4690199e-01	 1.2393597e+00	 1.4381131e-01	[ 1.1797362e+00]	 1.9532633e-01


.. parsed-literal::

      66	 1.1965748e+00	 1.4676514e-01	 1.2457227e+00	 1.4328011e-01	[ 1.1845934e+00]	 2.0814347e-01


.. parsed-literal::

      67	 1.2010344e+00	 1.4709467e-01	 1.2500777e+00	 1.4357715e-01	[ 1.1881734e+00]	 2.0922613e-01


.. parsed-literal::

      68	 1.2062142e+00	 1.4702860e-01	 1.2554580e+00	 1.4347186e-01	[ 1.1908030e+00]	 2.1902394e-01


.. parsed-literal::

      69	 1.2129599e+00	 1.4689051e-01	 1.2624456e+00	 1.4363035e-01	[ 1.1948695e+00]	 2.1195936e-01


.. parsed-literal::

      70	 1.2208497e+00	 1.4592215e-01	 1.2707993e+00	 1.4333862e-01	[ 1.2030143e+00]	 2.1942329e-01
      71	 1.2271352e+00	 1.4551377e-01	 1.2770942e+00	 1.4313213e-01	[ 1.2111418e+00]	 1.8428493e-01


.. parsed-literal::

      72	 1.2319314e+00	 1.4518069e-01	 1.2817804e+00	 1.4269359e-01	[ 1.2163911e+00]	 2.0199537e-01


.. parsed-literal::

      73	 1.2392061e+00	 1.4464048e-01	 1.2893163e+00	 1.4203830e-01	[ 1.2208758e+00]	 2.0138121e-01


.. parsed-literal::

      74	 1.2424444e+00	 1.4418428e-01	 1.2929405e+00	 1.4070353e-01	[ 1.2225063e+00]	 2.1356893e-01
      75	 1.2481364e+00	 1.4436709e-01	 1.2984988e+00	 1.4121300e-01	[ 1.2295197e+00]	 1.8291736e-01


.. parsed-literal::

      76	 1.2512786e+00	 1.4420196e-01	 1.3017287e+00	 1.4116686e-01	[ 1.2327689e+00]	 2.0788765e-01


.. parsed-literal::

      77	 1.2562363e+00	 1.4385736e-01	 1.3067677e+00	 1.4067972e-01	[ 1.2398821e+00]	 2.1046209e-01


.. parsed-literal::

      78	 1.2624192e+00	 1.4228680e-01	 1.3131902e+00	 1.3907451e-01	[ 1.2450219e+00]	 2.1506810e-01


.. parsed-literal::

      79	 1.2685504e+00	 1.4186611e-01	 1.3193265e+00	 1.3826570e-01	[ 1.2545593e+00]	 2.1721196e-01


.. parsed-literal::

      80	 1.2724040e+00	 1.4142992e-01	 1.3231387e+00	 1.3775167e-01	[ 1.2566171e+00]	 2.0814943e-01


.. parsed-literal::

      81	 1.2796968e+00	 1.4027142e-01	 1.3306622e+00	 1.3662176e-01	[ 1.2583839e+00]	 2.1241736e-01
      82	 1.2837803e+00	 1.3938522e-01	 1.3351241e+00	 1.3534997e-01	  1.2535971e+00 	 2.0741534e-01


.. parsed-literal::

      83	 1.2905323e+00	 1.3888617e-01	 1.3418290e+00	 1.3496651e-01	[ 1.2594634e+00]	 2.0819283e-01


.. parsed-literal::

      84	 1.2943382e+00	 1.3878948e-01	 1.3455509e+00	 1.3480784e-01	[ 1.2635282e+00]	 2.0839739e-01
      85	 1.2989752e+00	 1.3862957e-01	 1.3502289e+00	 1.3442821e-01	[ 1.2656543e+00]	 1.8904543e-01


.. parsed-literal::

      86	 1.3060144e+00	 1.3861867e-01	 1.3574709e+00	 1.3408007e-01	  1.2649374e+00 	 2.0552254e-01
      87	 1.3116141e+00	 1.3899823e-01	 1.3632254e+00	 1.3384843e-01	[ 1.2661311e+00]	 1.9894147e-01


.. parsed-literal::

      88	 1.3157963e+00	 1.3880351e-01	 1.3673476e+00	 1.3371508e-01	[ 1.2689682e+00]	 2.1915889e-01


.. parsed-literal::

      89	 1.3196945e+00	 1.3850570e-01	 1.3713695e+00	 1.3336715e-01	[ 1.2718262e+00]	 2.0967245e-01


.. parsed-literal::

      90	 1.3224636e+00	 1.3852944e-01	 1.3742912e+00	 1.3321366e-01	  1.2714780e+00 	 2.1371245e-01


.. parsed-literal::

      91	 1.3254960e+00	 1.3852039e-01	 1.3773588e+00	 1.3301732e-01	[ 1.2753647e+00]	 2.0491242e-01


.. parsed-literal::

      92	 1.3306621e+00	 1.3845492e-01	 1.3826268e+00	 1.3251425e-01	[ 1.2796591e+00]	 2.0667195e-01
      93	 1.3337839e+00	 1.3860599e-01	 1.3857399e+00	 1.3233579e-01	[ 1.2849178e+00]	 1.9983125e-01


.. parsed-literal::

      94	 1.3376482e+00	 1.3850448e-01	 1.3895455e+00	 1.3208830e-01	[ 1.2892951e+00]	 2.0956182e-01
      95	 1.3427299e+00	 1.3834765e-01	 1.3946583e+00	 1.3185968e-01	[ 1.2949811e+00]	 1.8560553e-01


.. parsed-literal::

      96	 1.3461869e+00	 1.3831121e-01	 1.3982086e+00	 1.3165190e-01	[ 1.2981598e+00]	 2.0909142e-01


.. parsed-literal::

      97	 1.3495085e+00	 1.3818092e-01	 1.4015436e+00	 1.3169033e-01	[ 1.2993150e+00]	 2.1006155e-01


.. parsed-literal::

      98	 1.3533157e+00	 1.3796803e-01	 1.4054647e+00	 1.3172711e-01	  1.2989841e+00 	 2.0673323e-01
      99	 1.3563447e+00	 1.3772073e-01	 1.4085443e+00	 1.3165145e-01	  1.2992700e+00 	 1.8716455e-01


.. parsed-literal::

     100	 1.3597904e+00	 1.3708940e-01	 1.4123106e+00	 1.3140487e-01	  1.2925934e+00 	 2.1257305e-01
     101	 1.3650581e+00	 1.3685446e-01	 1.4175072e+00	 1.3110553e-01	  1.2986937e+00 	 1.9853115e-01


.. parsed-literal::

     102	 1.3667996e+00	 1.3679937e-01	 1.4191710e+00	 1.3092678e-01	[ 1.3017616e+00]	 2.0908618e-01
     103	 1.3702045e+00	 1.3670493e-01	 1.4226409e+00	 1.3062098e-01	[ 1.3040046e+00]	 2.0197129e-01


.. parsed-literal::

     104	 1.3726767e+00	 1.3652647e-01	 1.4254674e+00	 1.3011321e-01	  1.2993240e+00 	 1.6843104e-01


.. parsed-literal::

     105	 1.3767337e+00	 1.3645158e-01	 1.4294665e+00	 1.3014817e-01	  1.3015254e+00 	 2.1086812e-01


.. parsed-literal::

     106	 1.3795875e+00	 1.3642785e-01	 1.4323806e+00	 1.3025266e-01	  1.2992657e+00 	 2.0654774e-01
     107	 1.3816614e+00	 1.3632000e-01	 1.4345400e+00	 1.3024249e-01	  1.2974521e+00 	 1.7220092e-01


.. parsed-literal::

     108	 1.3841602e+00	 1.3629977e-01	 1.4372821e+00	 1.3027520e-01	  1.2894260e+00 	 2.2290730e-01


.. parsed-literal::

     109	 1.3881072e+00	 1.3590655e-01	 1.4412374e+00	 1.3003642e-01	  1.2932168e+00 	 2.0995474e-01


.. parsed-literal::

     110	 1.3901697e+00	 1.3568378e-01	 1.4433005e+00	 1.2984418e-01	  1.2962381e+00 	 2.0448899e-01


.. parsed-literal::

     111	 1.3922545e+00	 1.3551328e-01	 1.4454716e+00	 1.2972976e-01	  1.2964719e+00 	 2.1149087e-01


.. parsed-literal::

     112	 1.3964033e+00	 1.3546198e-01	 1.4497640e+00	 1.2953318e-01	  1.2952639e+00 	 2.1074891e-01


.. parsed-literal::

     113	 1.3993175e+00	 1.3557230e-01	 1.4528565e+00	 1.2958194e-01	  1.2973851e+00 	 3.4727049e-01


.. parsed-literal::

     114	 1.4025282e+00	 1.3575065e-01	 1.4561282e+00	 1.2943037e-01	  1.2963955e+00 	 2.1232986e-01


.. parsed-literal::

     115	 1.4051873e+00	 1.3591392e-01	 1.4587934e+00	 1.2923186e-01	  1.2987251e+00 	 2.1611023e-01


.. parsed-literal::

     116	 1.4083646e+00	 1.3616431e-01	 1.4620517e+00	 1.2892474e-01	[ 1.3040493e+00]	 2.1174765e-01


.. parsed-literal::

     117	 1.4107742e+00	 1.3629141e-01	 1.4645265e+00	 1.2857721e-01	[ 1.3065614e+00]	 2.2069597e-01


.. parsed-literal::

     118	 1.4125150e+00	 1.3618084e-01	 1.4662402e+00	 1.2845158e-01	[ 1.3080925e+00]	 2.0404267e-01


.. parsed-literal::

     119	 1.4148878e+00	 1.3602848e-01	 1.4686632e+00	 1.2825171e-01	  1.3078928e+00 	 2.0439315e-01
     120	 1.4172098e+00	 1.3592487e-01	 1.4710341e+00	 1.2795206e-01	[ 1.3081909e+00]	 1.9859362e-01


.. parsed-literal::

     121	 1.4194678e+00	 1.3588455e-01	 1.4736027e+00	 1.2767203e-01	  1.3071164e+00 	 2.0697427e-01


.. parsed-literal::

     122	 1.4230857e+00	 1.3561114e-01	 1.4770221e+00	 1.2726720e-01	[ 1.3116953e+00]	 2.2120309e-01


.. parsed-literal::

     123	 1.4242813e+00	 1.3561818e-01	 1.4781779e+00	 1.2722317e-01	[ 1.3140805e+00]	 2.1371794e-01


.. parsed-literal::

     124	 1.4265988e+00	 1.3537075e-01	 1.4804876e+00	 1.2699211e-01	[ 1.3168330e+00]	 2.0820928e-01


.. parsed-literal::

     125	 1.4284666e+00	 1.3500822e-01	 1.4825101e+00	 1.2676149e-01	[ 1.3234380e+00]	 2.1090722e-01


.. parsed-literal::

     126	 1.4306081e+00	 1.3476432e-01	 1.4846047e+00	 1.2669189e-01	  1.3213740e+00 	 2.1465468e-01


.. parsed-literal::

     127	 1.4326184e+00	 1.3446348e-01	 1.4866259e+00	 1.2663302e-01	  1.3204220e+00 	 2.1533704e-01


.. parsed-literal::

     128	 1.4349667e+00	 1.3409337e-01	 1.4890271e+00	 1.2666407e-01	  1.3199114e+00 	 2.1511006e-01


.. parsed-literal::

     129	 1.4367263e+00	 1.3393097e-01	 1.4909738e+00	 1.2717003e-01	  1.3202200e+00 	 2.0687532e-01
     130	 1.4392035e+00	 1.3379478e-01	 1.4933374e+00	 1.2698798e-01	  1.3221929e+00 	 1.9037843e-01


.. parsed-literal::

     131	 1.4406794e+00	 1.3378763e-01	 1.4947888e+00	 1.2690346e-01	  1.3231508e+00 	 2.1008635e-01


.. parsed-literal::

     132	 1.4426672e+00	 1.3373847e-01	 1.4968114e+00	 1.2678074e-01	  1.3224936e+00 	 2.1984744e-01


.. parsed-literal::

     133	 1.4449798e+00	 1.3363238e-01	 1.4993136e+00	 1.2666247e-01	  1.3158327e+00 	 2.0912147e-01
     134	 1.4478299e+00	 1.3349732e-01	 1.5021848e+00	 1.2647413e-01	  1.3137339e+00 	 1.9632268e-01


.. parsed-literal::

     135	 1.4496408e+00	 1.3341992e-01	 1.5039672e+00	 1.2625875e-01	  1.3133073e+00 	 2.0901799e-01


.. parsed-literal::

     136	 1.4515871e+00	 1.3324168e-01	 1.5059885e+00	 1.2617812e-01	  1.3087949e+00 	 2.0283794e-01


.. parsed-literal::

     137	 1.4532298e+00	 1.3317277e-01	 1.5078742e+00	 1.2586724e-01	  1.3073939e+00 	 2.0552659e-01
     138	 1.4555441e+00	 1.3309945e-01	 1.5101339e+00	 1.2587761e-01	  1.3045102e+00 	 1.9857717e-01


.. parsed-literal::

     139	 1.4572838e+00	 1.3310171e-01	 1.5119191e+00	 1.2590701e-01	  1.3020504e+00 	 2.0104122e-01


.. parsed-literal::

     140	 1.4586968e+00	 1.3314144e-01	 1.5133439e+00	 1.2583611e-01	  1.3011824e+00 	 2.0949364e-01


.. parsed-literal::

     141	 1.4594992e+00	 1.3322856e-01	 1.5143182e+00	 1.2594096e-01	  1.3053825e+00 	 2.1277881e-01
     142	 1.4616197e+00	 1.3314095e-01	 1.5163324e+00	 1.2572109e-01	  1.3049108e+00 	 2.0279336e-01


.. parsed-literal::

     143	 1.4625772e+00	 1.3306161e-01	 1.5172732e+00	 1.2563333e-01	  1.3047291e+00 	 2.0354557e-01


.. parsed-literal::

     144	 1.4640559e+00	 1.3289011e-01	 1.5187857e+00	 1.2551701e-01	  1.3047390e+00 	 2.1086192e-01


.. parsed-literal::

     145	 1.4666034e+00	 1.3259373e-01	 1.5214411e+00	 1.2540729e-01	  1.3032590e+00 	 2.0794153e-01


.. parsed-literal::

     146	 1.4680982e+00	 1.3239529e-01	 1.5231140e+00	 1.2542038e-01	  1.3011826e+00 	 3.4015894e-01


.. parsed-literal::

     147	 1.4705121e+00	 1.3218117e-01	 1.5256543e+00	 1.2551696e-01	  1.2983061e+00 	 2.2032094e-01


.. parsed-literal::

     148	 1.4718516e+00	 1.3217481e-01	 1.5270576e+00	 1.2568250e-01	  1.2958216e+00 	 2.1855903e-01
     149	 1.4736563e+00	 1.3217036e-01	 1.5289938e+00	 1.2601819e-01	  1.2886585e+00 	 1.8755817e-01


.. parsed-literal::

     150	 1.4749523e+00	 1.3223226e-01	 1.5304509e+00	 1.2626803e-01	  1.2865763e+00 	 2.0461965e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.12 s, total: 2min 7s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f5f613afaf0>



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
    CPU times: user 1.9 s, sys: 46 ms, total: 1.95 s
    Wall time: 665 ms


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

