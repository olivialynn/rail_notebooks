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
       1	-3.2993053e-01	 3.1688238e-01	-3.2026521e-01	 3.3569814e-01	[-3.5810491e-01]	 4.7019076e-01


.. parsed-literal::

       2	-2.6183531e-01	 3.0670732e-01	-2.3812512e-01	 3.2448241e-01	[-2.9527578e-01]	 2.3136449e-01


.. parsed-literal::

       3	-2.1947607e-01	 2.8709423e-01	-1.7815810e-01	 3.0228807e-01	[-2.4107212e-01]	 2.7969074e-01


.. parsed-literal::

       4	-1.8237468e-01	 2.6419283e-01	-1.4100772e-01	 2.7322322e-01	[-1.9366167e-01]	 2.0697904e-01


.. parsed-literal::

       5	-9.3442864e-02	 2.5390114e-01	-5.8671485e-02	 2.6597829e-01	[-1.1512495e-01]	 2.0914841e-01


.. parsed-literal::

       6	-6.0710790e-02	 2.4965743e-01	-3.0640502e-02	 2.6362102e-01	[-8.1729799e-02]	 2.1172118e-01
       7	-4.1790935e-02	 2.4622801e-01	-1.7781175e-02	 2.5897603e-01	[-6.8611972e-02]	 1.8897939e-01


.. parsed-literal::

       8	-3.0927545e-02	 2.4449995e-01	-1.0563299e-02	 2.5735474e-01	[-6.3325295e-02]	 2.0907354e-01


.. parsed-literal::

       9	-1.8641290e-02	 2.4226513e-01	-1.0674576e-03	 2.5538906e-01	[-5.3568586e-02]	 2.0177007e-01


.. parsed-literal::

      10	-1.1386080e-02	 2.4041009e-01	 3.7560646e-03	 2.5363777e-01	 -5.6621627e-02 	 2.0237994e-01


.. parsed-literal::

      11	-1.6288286e-03	 2.3913853e-01	 1.2795476e-02	 2.5222750e-01	[-3.8586970e-02]	 2.1152711e-01


.. parsed-literal::

      12	 7.4650974e-04	 2.3875168e-01	 1.4849709e-02	 2.5152120e-01	[-3.5921227e-02]	 2.1147299e-01


.. parsed-literal::

      13	 5.0981182e-03	 2.3804049e-01	 1.8829895e-02	 2.5031182e-01	[-2.9443181e-02]	 2.1119785e-01


.. parsed-literal::

      14	 1.4997053e-01	 2.2199088e-01	 1.7328049e-01	 2.3652450e-01	[ 1.4907191e-01]	 4.4335079e-01


.. parsed-literal::

      15	 2.1716626e-01	 2.1691161e-01	 2.4179667e-01	 2.2876003e-01	[ 2.2358690e-01]	 2.0707989e-01


.. parsed-literal::

      16	 3.0622668e-01	 2.1326062e-01	 3.3526022e-01	 2.2365267e-01	[ 3.1298168e-01]	 2.1040726e-01


.. parsed-literal::

      17	 3.5282473e-01	 2.1090353e-01	 3.8446713e-01	 2.2386489e-01	[ 3.5332523e-01]	 2.2272944e-01


.. parsed-literal::

      18	 4.0012613e-01	 2.0886560e-01	 4.3264858e-01	 2.2161957e-01	[ 4.0289972e-01]	 2.1049118e-01


.. parsed-literal::

      19	 4.8075841e-01	 2.0593184e-01	 5.1485537e-01	 2.1609647e-01	[ 4.9120221e-01]	 2.1009994e-01


.. parsed-literal::

      20	 5.8998055e-01	 1.9895230e-01	 6.2806790e-01	 2.0643190e-01	[ 6.0569864e-01]	 2.1682572e-01


.. parsed-literal::

      21	 6.1959110e-01	 2.0056723e-01	 6.5985048e-01	 2.1356231e-01	[ 6.2640657e-01]	 2.1861696e-01


.. parsed-literal::

      22	 6.5698052e-01	 1.9436997e-01	 6.9439911e-01	 2.0563888e-01	[ 6.6796229e-01]	 2.1280980e-01
      23	 6.9829133e-01	 1.9124445e-01	 7.3641920e-01	 2.0281636e-01	[ 7.0563474e-01]	 1.9261932e-01


.. parsed-literal::

      24	 7.1854503e-01	 1.9507344e-01	 7.5716805e-01	 2.0772398e-01	[ 7.2468828e-01]	 1.9038010e-01


.. parsed-literal::

      25	 7.5309030e-01	 1.8844743e-01	 7.9172284e-01	 2.0149841e-01	[ 7.5381985e-01]	 2.1378255e-01


.. parsed-literal::

      26	 7.7795872e-01	 1.8857494e-01	 8.1650399e-01	 2.0306986e-01	[ 7.7319720e-01]	 2.1826673e-01


.. parsed-literal::

      27	 8.0677781e-01	 1.9488103e-01	 8.4557032e-01	 2.1116758e-01	[ 8.0000436e-01]	 2.1325612e-01
      28	 8.3585551e-01	 1.8922421e-01	 8.7551404e-01	 2.0513414e-01	[ 8.3009811e-01]	 1.7816281e-01


.. parsed-literal::

      29	 8.5252068e-01	 1.8579704e-01	 8.9238158e-01	 2.0052791e-01	[ 8.4390890e-01]	 2.1704960e-01
      30	 8.7191533e-01	 1.8497001e-01	 9.1192789e-01	 1.9878032e-01	[ 8.6175894e-01]	 1.8401647e-01


.. parsed-literal::

      31	 8.9338919e-01	 1.8340319e-01	 9.3425389e-01	 1.9744442e-01	[ 8.9263762e-01]	 2.0907259e-01
      32	 9.1497541e-01	 1.8238471e-01	 9.5644164e-01	 1.9622875e-01	[ 9.1393958e-01]	 2.0130777e-01


.. parsed-literal::

      33	 9.3537287e-01	 1.8178436e-01	 9.7748284e-01	 1.9549637e-01	[ 9.3703758e-01]	 1.8048596e-01
      34	 9.5037767e-01	 1.7997087e-01	 9.9292674e-01	 1.9308763e-01	[ 9.4835133e-01]	 1.7524004e-01


.. parsed-literal::

      35	 9.6482550e-01	 1.7812185e-01	 1.0079829e+00	 1.9134621e-01	[ 9.5881827e-01]	 2.0824671e-01


.. parsed-literal::

      36	 9.8130426e-01	 1.7783184e-01	 1.0252496e+00	 1.9306645e-01	[ 9.6635235e-01]	 2.1049571e-01


.. parsed-literal::

      37	 9.9722706e-01	 1.7650791e-01	 1.0415781e+00	 1.9085915e-01	[ 9.7759802e-01]	 2.0649815e-01
      38	 1.0102940e+00	 1.7565556e-01	 1.0549706e+00	 1.8931433e-01	[ 9.8508787e-01]	 1.6603756e-01


.. parsed-literal::

      39	 1.0309240e+00	 1.7605504e-01	 1.0768617e+00	 1.8988012e-01	[ 9.9001716e-01]	 2.1006393e-01


.. parsed-literal::

      40	 1.0392068e+00	 1.7329360e-01	 1.0860901e+00	 1.8659466e-01	  9.7981472e-01 	 2.0685768e-01


.. parsed-literal::

      41	 1.0521868e+00	 1.7316310e-01	 1.0987099e+00	 1.8667693e-01	[ 9.9542993e-01]	 2.1163344e-01
      42	 1.0587471e+00	 1.7289117e-01	 1.1053196e+00	 1.8650503e-01	[ 9.9944400e-01]	 1.9771600e-01


.. parsed-literal::

      43	 1.0671358e+00	 1.7257761e-01	 1.1140673e+00	 1.8605854e-01	[ 9.9966884e-01]	 2.0771241e-01
      44	 1.0797039e+00	 1.7134138e-01	 1.1266041e+00	 1.8437652e-01	[ 1.0070871e+00]	 1.8769598e-01


.. parsed-literal::

      45	 1.0887226e+00	 1.7022051e-01	 1.1358675e+00	 1.8279733e-01	  1.0044342e+00 	 2.1902990e-01


.. parsed-literal::

      46	 1.0970222e+00	 1.6902499e-01	 1.1439637e+00	 1.8102091e-01	[ 1.0173660e+00]	 2.1692085e-01
      47	 1.1084096e+00	 1.6719032e-01	 1.1553289e+00	 1.7835223e-01	[ 1.0373974e+00]	 1.9917488e-01


.. parsed-literal::

      48	 1.1183616e+00	 1.6641892e-01	 1.1653250e+00	 1.7741460e-01	[ 1.0536879e+00]	 2.0596766e-01


.. parsed-literal::

      49	 1.1304227e+00	 1.6283540e-01	 1.1777809e+00	 1.7367768e-01	[ 1.0754738e+00]	 2.0818400e-01
      50	 1.1416454e+00	 1.6352922e-01	 1.1890387e+00	 1.7453400e-01	[ 1.0950904e+00]	 1.7691922e-01


.. parsed-literal::

      51	 1.1489887e+00	 1.6357070e-01	 1.1963295e+00	 1.7564133e-01	[ 1.1022420e+00]	 2.0992613e-01


.. parsed-literal::

      52	 1.1625777e+00	 1.6248528e-01	 1.2105161e+00	 1.7601779e-01	[ 1.1107120e+00]	 2.0484447e-01


.. parsed-literal::

      53	 1.1714624e+00	 1.6253841e-01	 1.2197034e+00	 1.7702555e-01	[ 1.1164754e+00]	 2.1194530e-01


.. parsed-literal::

      54	 1.1814827e+00	 1.6048883e-01	 1.2297201e+00	 1.7480319e-01	[ 1.1301247e+00]	 2.1321177e-01
      55	 1.1881932e+00	 1.5914672e-01	 1.2365588e+00	 1.7334493e-01	[ 1.1368522e+00]	 2.0267653e-01


.. parsed-literal::

      56	 1.1975606e+00	 1.5786898e-01	 1.2462795e+00	 1.7225607e-01	[ 1.1419368e+00]	 1.8903041e-01


.. parsed-literal::

      57	 1.2084126e+00	 1.5627575e-01	 1.2576148e+00	 1.7220104e-01	[ 1.1441982e+00]	 2.1231151e-01


.. parsed-literal::

      58	 1.2179343e+00	 1.5555925e-01	 1.2673225e+00	 1.7228543e-01	[ 1.1443231e+00]	 2.0302343e-01


.. parsed-literal::

      59	 1.2250518e+00	 1.5548734e-01	 1.2741087e+00	 1.7213551e-01	[ 1.1520143e+00]	 2.1800947e-01


.. parsed-literal::

      60	 1.2339798e+00	 1.5499808e-01	 1.2832957e+00	 1.7257825e-01	[ 1.1576416e+00]	 2.1334243e-01
      61	 1.2418125e+00	 1.5453321e-01	 1.2913174e+00	 1.7210655e-01	[ 1.1576923e+00]	 1.9653583e-01


.. parsed-literal::

      62	 1.2489902e+00	 1.5405770e-01	 1.2986496e+00	 1.7188633e-01	[ 1.1601742e+00]	 2.0977926e-01


.. parsed-literal::

      63	 1.2577334e+00	 1.5341147e-01	 1.3076517e+00	 1.7139408e-01	  1.1568228e+00 	 2.0402718e-01


.. parsed-literal::

      64	 1.2647957e+00	 1.5215071e-01	 1.3149910e+00	 1.7008061e-01	  1.1524741e+00 	 2.2120118e-01


.. parsed-literal::

      65	 1.2730124e+00	 1.5090676e-01	 1.3233498e+00	 1.6839403e-01	  1.1487709e+00 	 2.0355153e-01
      66	 1.2822017e+00	 1.4928451e-01	 1.3327466e+00	 1.6641800e-01	  1.1320333e+00 	 2.0341063e-01


.. parsed-literal::

      67	 1.2888562e+00	 1.4772843e-01	 1.3397524e+00	 1.6490082e-01	  1.1165718e+00 	 2.0573473e-01


.. parsed-literal::

      68	 1.2957768e+00	 1.4723440e-01	 1.3465637e+00	 1.6470685e-01	  1.1177770e+00 	 2.1049380e-01
      69	 1.3033739e+00	 1.4609503e-01	 1.3543555e+00	 1.6391556e-01	  1.1016254e+00 	 1.9288445e-01


.. parsed-literal::

      70	 1.3107619e+00	 1.4547367e-01	 1.3618149e+00	 1.6314649e-01	  1.0942651e+00 	 2.1443605e-01


.. parsed-literal::

      71	 1.3159139e+00	 1.4356697e-01	 1.3671340e+00	 1.6058110e-01	  1.0856279e+00 	 2.1048379e-01


.. parsed-literal::

      72	 1.3229229e+00	 1.4337276e-01	 1.3739799e+00	 1.5988056e-01	  1.0886005e+00 	 2.1564722e-01


.. parsed-literal::

      73	 1.3270480e+00	 1.4322802e-01	 1.3780230e+00	 1.5926142e-01	  1.0968914e+00 	 2.1848631e-01


.. parsed-literal::

      74	 1.3332531e+00	 1.4223544e-01	 1.3843366e+00	 1.5763726e-01	  1.1012619e+00 	 2.1340656e-01
      75	 1.3386814e+00	 1.4221048e-01	 1.3899589e+00	 1.5733855e-01	  1.1137819e+00 	 1.8419290e-01


.. parsed-literal::

      76	 1.3452036e+00	 1.4114314e-01	 1.3963690e+00	 1.5663360e-01	  1.1213267e+00 	 2.0259380e-01


.. parsed-literal::

      77	 1.3494014e+00	 1.4056863e-01	 1.4006332e+00	 1.5665604e-01	  1.1214537e+00 	 2.0539856e-01
      78	 1.3545373e+00	 1.4011170e-01	 1.4058872e+00	 1.5662141e-01	  1.1265651e+00 	 1.8106103e-01


.. parsed-literal::

      79	 1.3598953e+00	 1.3933085e-01	 1.4116923e+00	 1.5702842e-01	  1.1325605e+00 	 2.0948124e-01


.. parsed-literal::

      80	 1.3659918e+00	 1.3916272e-01	 1.4177375e+00	 1.5641656e-01	  1.1413331e+00 	 2.0857310e-01


.. parsed-literal::

      81	 1.3696798e+00	 1.3893242e-01	 1.4213565e+00	 1.5600461e-01	  1.1455433e+00 	 2.1384811e-01
      82	 1.3752265e+00	 1.3799247e-01	 1.4270898e+00	 1.5509763e-01	  1.1564402e+00 	 1.7919564e-01


.. parsed-literal::

      83	 1.3783754e+00	 1.3719197e-01	 1.4304818e+00	 1.5469770e-01	  1.1504264e+00 	 2.1553707e-01
      84	 1.3828234e+00	 1.3696476e-01	 1.4348401e+00	 1.5464440e-01	  1.1568732e+00 	 1.9479966e-01


.. parsed-literal::

      85	 1.3872605e+00	 1.3629904e-01	 1.4393956e+00	 1.5441950e-01	[ 1.1628586e+00]	 2.1157265e-01


.. parsed-literal::

      86	 1.3908889e+00	 1.3569230e-01	 1.4431142e+00	 1.5399266e-01	[ 1.1661616e+00]	 2.1854043e-01


.. parsed-literal::

      87	 1.3950421e+00	 1.3391972e-01	 1.4476035e+00	 1.5268492e-01	[ 1.1738989e+00]	 2.1161580e-01


.. parsed-literal::

      88	 1.4000611e+00	 1.3350585e-01	 1.4526681e+00	 1.5184515e-01	[ 1.1801583e+00]	 2.0786023e-01


.. parsed-literal::

      89	 1.4027068e+00	 1.3352556e-01	 1.4551654e+00	 1.5144218e-01	[ 1.1810630e+00]	 2.0683622e-01


.. parsed-literal::

      90	 1.4072901e+00	 1.3297880e-01	 1.4597566e+00	 1.5037229e-01	[ 1.1831195e+00]	 2.0829463e-01
      91	 1.4108785e+00	 1.3273595e-01	 1.4634756e+00	 1.4922450e-01	[ 1.1863051e+00]	 1.8363404e-01


.. parsed-literal::

      92	 1.4155847e+00	 1.3231028e-01	 1.4680735e+00	 1.4870590e-01	[ 1.1925291e+00]	 2.1935964e-01


.. parsed-literal::

      93	 1.4183323e+00	 1.3203615e-01	 1.4707851e+00	 1.4862821e-01	[ 1.1987645e+00]	 2.1140099e-01


.. parsed-literal::

      94	 1.4216162e+00	 1.3175966e-01	 1.4741032e+00	 1.4831601e-01	[ 1.2090971e+00]	 2.1165419e-01
      95	 1.4240653e+00	 1.3144303e-01	 1.4766429e+00	 1.4841124e-01	[ 1.2145917e+00]	 1.9948077e-01


.. parsed-literal::

      96	 1.4279172e+00	 1.3142926e-01	 1.4804267e+00	 1.4806730e-01	[ 1.2238565e+00]	 2.0808482e-01


.. parsed-literal::

      97	 1.4306659e+00	 1.3116348e-01	 1.4832436e+00	 1.4757611e-01	[ 1.2272196e+00]	 2.1342802e-01
      98	 1.4335881e+00	 1.3093963e-01	 1.4862427e+00	 1.4753710e-01	[ 1.2296450e+00]	 1.8346000e-01


.. parsed-literal::

      99	 1.4362152e+00	 1.3073934e-01	 1.4890584e+00	 1.4802228e-01	  1.2259396e+00 	 2.0513368e-01


.. parsed-literal::

     100	 1.4401984e+00	 1.3056229e-01	 1.4930046e+00	 1.4793750e-01	[ 1.2319542e+00]	 2.0298290e-01


.. parsed-literal::

     101	 1.4420240e+00	 1.3049672e-01	 1.4947813e+00	 1.4817416e-01	[ 1.2333049e+00]	 2.0171595e-01


.. parsed-literal::

     102	 1.4445635e+00	 1.3028715e-01	 1.4973305e+00	 1.4832569e-01	[ 1.2350961e+00]	 2.0632577e-01


.. parsed-literal::

     103	 1.4482070e+00	 1.2979749e-01	 1.5010573e+00	 1.4845005e-01	  1.2349438e+00 	 2.1410060e-01
     104	 1.4509632e+00	 1.2914608e-01	 1.5039875e+00	 1.4834344e-01	  1.2332384e+00 	 1.7567945e-01


.. parsed-literal::

     105	 1.4539317e+00	 1.2896345e-01	 1.5068934e+00	 1.4810348e-01	  1.2344438e+00 	 2.1363401e-01


.. parsed-literal::

     106	 1.4561061e+00	 1.2868402e-01	 1.5090911e+00	 1.4783973e-01	  1.2346745e+00 	 2.1109033e-01
     107	 1.4584097e+00	 1.2841051e-01	 1.5114366e+00	 1.4768874e-01	  1.2329038e+00 	 2.0435333e-01


.. parsed-literal::

     108	 1.4604151e+00	 1.2779879e-01	 1.5136822e+00	 1.4756013e-01	[ 1.2361363e+00]	 2.0006871e-01
     109	 1.4648289e+00	 1.2776259e-01	 1.5179866e+00	 1.4772037e-01	  1.2321155e+00 	 1.9852352e-01


.. parsed-literal::

     110	 1.4663898e+00	 1.2780101e-01	 1.5195123e+00	 1.4789888e-01	  1.2332692e+00 	 2.0959091e-01
     111	 1.4694359e+00	 1.2771534e-01	 1.5226146e+00	 1.4826400e-01	  1.2326013e+00 	 1.7853975e-01


.. parsed-literal::

     112	 1.4710311e+00	 1.2789865e-01	 1.5243774e+00	 1.4843622e-01	  1.2342746e+00 	 2.1250176e-01


.. parsed-literal::

     113	 1.4744026e+00	 1.2763021e-01	 1.5277450e+00	 1.4863217e-01	  1.2320791e+00 	 2.1022582e-01
     114	 1.4760436e+00	 1.2739141e-01	 1.5294088e+00	 1.4846848e-01	  1.2316670e+00 	 1.9690800e-01


.. parsed-literal::

     115	 1.4788372e+00	 1.2708247e-01	 1.5322746e+00	 1.4833529e-01	  1.2300952e+00 	 2.1032548e-01
     116	 1.4800990e+00	 1.2677647e-01	 1.5337255e+00	 1.4777204e-01	  1.2299588e+00 	 1.9851923e-01


.. parsed-literal::

     117	 1.4834727e+00	 1.2667874e-01	 1.5370038e+00	 1.4780248e-01	  1.2299358e+00 	 1.9564652e-01


.. parsed-literal::

     118	 1.4849659e+00	 1.2663227e-01	 1.5385036e+00	 1.4784358e-01	  1.2297626e+00 	 2.1449518e-01
     119	 1.4869174e+00	 1.2651535e-01	 1.5404908e+00	 1.4781477e-01	  1.2294071e+00 	 2.0254278e-01


.. parsed-literal::

     120	 1.4903683e+00	 1.2608289e-01	 1.5440429e+00	 1.4749353e-01	  1.2254614e+00 	 2.1601439e-01


.. parsed-literal::

     121	 1.4920312e+00	 1.2549384e-01	 1.5459573e+00	 1.4752672e-01	  1.2105829e+00 	 2.1397281e-01


.. parsed-literal::

     122	 1.4959943e+00	 1.2524388e-01	 1.5497688e+00	 1.4691636e-01	  1.2177201e+00 	 2.0693707e-01


.. parsed-literal::

     123	 1.4972778e+00	 1.2513440e-01	 1.5509780e+00	 1.4669554e-01	  1.2189963e+00 	 2.0362210e-01


.. parsed-literal::

     124	 1.4994963e+00	 1.2482212e-01	 1.5531117e+00	 1.4635098e-01	  1.2208611e+00 	 2.1006680e-01


.. parsed-literal::

     125	 1.5020343e+00	 1.2433809e-01	 1.5556137e+00	 1.4611232e-01	  1.2177736e+00 	 2.0870233e-01


.. parsed-literal::

     126	 1.5043418e+00	 1.2418780e-01	 1.5578571e+00	 1.4617630e-01	  1.2211290e+00 	 2.1209073e-01


.. parsed-literal::

     127	 1.5062289e+00	 1.2398915e-01	 1.5597498e+00	 1.4636852e-01	  1.2235871e+00 	 2.1301913e-01
     128	 1.5075819e+00	 1.2389201e-01	 1.5611119e+00	 1.4652948e-01	  1.2239791e+00 	 1.9917631e-01


.. parsed-literal::

     129	 1.5093110e+00	 1.2367924e-01	 1.5629764e+00	 1.4659642e-01	  1.2293609e+00 	 2.0775342e-01


.. parsed-literal::

     130	 1.5114952e+00	 1.2359925e-01	 1.5651439e+00	 1.4671119e-01	  1.2230306e+00 	 2.0997143e-01


.. parsed-literal::

     131	 1.5123292e+00	 1.2358207e-01	 1.5659677e+00	 1.4649807e-01	  1.2225053e+00 	 2.1017218e-01
     132	 1.5137408e+00	 1.2349018e-01	 1.5674098e+00	 1.4618219e-01	  1.2206762e+00 	 1.6803026e-01


.. parsed-literal::

     133	 1.5150419e+00	 1.2333573e-01	 1.5688540e+00	 1.4577675e-01	  1.2144169e+00 	 2.1268320e-01


.. parsed-literal::

     134	 1.5169839e+00	 1.2326598e-01	 1.5707656e+00	 1.4568097e-01	  1.2161059e+00 	 2.0684290e-01


.. parsed-literal::

     135	 1.5181762e+00	 1.2318253e-01	 1.5719929e+00	 1.4566790e-01	  1.2163930e+00 	 2.1610332e-01
     136	 1.5190029e+00	 1.2313557e-01	 1.5728534e+00	 1.4563397e-01	  1.2159783e+00 	 1.9727755e-01


.. parsed-literal::

     137	 1.5212344e+00	 1.2303581e-01	 1.5751695e+00	 1.4547415e-01	  1.2159230e+00 	 2.0151305e-01


.. parsed-literal::

     138	 1.5222907e+00	 1.2309173e-01	 1.5762925e+00	 1.4552453e-01	  1.2109687e+00 	 3.2068300e-01


.. parsed-literal::

     139	 1.5240413e+00	 1.2305667e-01	 1.5781233e+00	 1.4536071e-01	  1.2126539e+00 	 2.1204233e-01


.. parsed-literal::

     140	 1.5253374e+00	 1.2304117e-01	 1.5794343e+00	 1.4532338e-01	  1.2140248e+00 	 2.0536280e-01
     141	 1.5274680e+00	 1.2300596e-01	 1.5815950e+00	 1.4542458e-01	  1.2147404e+00 	 2.0180511e-01


.. parsed-literal::

     142	 1.5285151e+00	 1.2301791e-01	 1.5826502e+00	 1.4540935e-01	  1.2182240e+00 	 1.9410872e-01


.. parsed-literal::

     143	 1.5301594e+00	 1.2289545e-01	 1.5842399e+00	 1.4545048e-01	  1.2160775e+00 	 2.0824981e-01
     144	 1.5313585e+00	 1.2280071e-01	 1.5854327e+00	 1.4544921e-01	  1.2159440e+00 	 1.7137861e-01


.. parsed-literal::

     145	 1.5326987e+00	 1.2267809e-01	 1.5868103e+00	 1.4545452e-01	  1.2156210e+00 	 1.7891169e-01


.. parsed-literal::

     146	 1.5347119e+00	 1.2248023e-01	 1.5888895e+00	 1.4538526e-01	  1.2185331e+00 	 2.1475959e-01
     147	 1.5365004e+00	 1.2230725e-01	 1.5907846e+00	 1.4547431e-01	  1.2188623e+00 	 1.8204689e-01


.. parsed-literal::

     148	 1.5376585e+00	 1.2224104e-01	 1.5919355e+00	 1.4547773e-01	  1.2189767e+00 	 1.9945645e-01


.. parsed-literal::

     149	 1.5387694e+00	 1.2216583e-01	 1.5930347e+00	 1.4544208e-01	  1.2223583e+00 	 2.1137738e-01


.. parsed-literal::

     150	 1.5399471e+00	 1.2209615e-01	 1.5942096e+00	 1.4539635e-01	  1.2201069e+00 	 2.0808530e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.15 s, total: 2min 6s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff7847fcc40>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.1 s, sys: 41.9 ms, total: 2.14 s
    Wall time: 642 ms


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

