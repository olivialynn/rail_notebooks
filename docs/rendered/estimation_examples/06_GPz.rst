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
       1	-3.3086441e-01	 3.1659362e-01	-3.2109454e-01	 3.3570954e-01	[-3.5957296e-01]	 4.6726418e-01


.. parsed-literal::

       2	-2.5483873e-01	 3.0365244e-01	-2.2856953e-01	 3.2575122e-01	[-2.9941749e-01]	 2.3065615e-01


.. parsed-literal::

       3	-2.1289815e-01	 2.8540479e-01	-1.7123653e-01	 3.1044146e-01	[-2.8253697e-01]	 2.8504062e-01
       4	-1.8014301e-01	 2.6036679e-01	-1.3812724e-01	 2.8776187e-01	 -3.1415581e-01 	 1.7321396e-01


.. parsed-literal::

       5	-9.3378779e-02	 2.5326794e-01	-5.5155991e-02	 2.7862193e-01	[-1.8554288e-01]	 2.0237875e-01


.. parsed-literal::

       6	-5.4853311e-02	 2.4662263e-01	-2.0224302e-02	 2.7149445e-01	[-1.1967962e-01]	 2.1081686e-01


.. parsed-literal::

       7	-3.6735442e-02	 2.4428249e-01	-9.9715999e-03	 2.6865284e-01	[-1.1368862e-01]	 2.0359588e-01


.. parsed-literal::

       8	-2.2200552e-02	 2.4196852e-01	-1.6823562e-04	 2.6576955e-01	[-1.0313495e-01]	 2.0934105e-01
       9	-5.4028793e-03	 2.3895501e-01	 1.2806210e-02	 2.6244925e-01	[-9.0771090e-02]	 1.8838143e-01


.. parsed-literal::

      10	 3.1035739e-03	 2.3786615e-01	 1.8549762e-02	 2.6196032e-01	[-9.0557576e-02]	 2.1554971e-01


.. parsed-literal::

      11	 1.0287705e-02	 2.3626660e-01	 2.4944062e-02	 2.6142590e-01	[-8.7828727e-02]	 2.1025133e-01


.. parsed-literal::

      12	 1.2543083e-02	 2.3583474e-01	 2.6964577e-02	 2.6111342e-01	[-8.5342013e-02]	 2.0922303e-01


.. parsed-literal::

      13	 1.7488697e-02	 2.3490655e-01	 3.1676968e-02	 2.6051167e-01	[-8.2440840e-02]	 2.0198512e-01


.. parsed-literal::

      14	 2.7174263e-02	 2.3266290e-01	 4.2908138e-02	 2.5937837e-01	[-6.9612331e-02]	 2.1182942e-01
      15	 3.7831594e-02	 2.2224270e-01	 5.5808163e-02	 2.4972176e-01	[-7.4147081e-03]	 1.7430615e-01


.. parsed-literal::

      16	 1.7126513e-01	 2.1728792e-01	 1.9433483e-01	 2.4497876e-01	[ 9.1435374e-02]	 2.0863128e-01


.. parsed-literal::

      17	 2.3782683e-01	 2.1151782e-01	 2.6727322e-01	 2.3916964e-01	[ 1.0329624e-01]	 2.1186280e-01


.. parsed-literal::

      18	 2.8131754e-01	 2.1100976e-01	 3.1170731e-01	 2.3689115e-01	[ 1.6916025e-01]	 2.1015143e-01


.. parsed-literal::

      19	 3.2132276e-01	 2.0535242e-01	 3.5117764e-01	 2.3227930e-01	[ 2.2258982e-01]	 2.0975924e-01
      20	 3.6332384e-01	 1.9827648e-01	 3.9465066e-01	 2.2526928e-01	[ 2.6941900e-01]	 1.8229151e-01


.. parsed-literal::

      21	 4.4677435e-01	 1.9179440e-01	 4.7946472e-01	 2.1843970e-01	[ 3.4592672e-01]	 1.9559312e-01


.. parsed-literal::

      22	 5.6154139e-01	 1.9347493e-01	 5.9525010e-01	 2.1722694e-01	[ 4.5908187e-01]	 2.1027160e-01
      23	 6.3001331e-01	 1.9641364e-01	 6.6605068e-01	 2.2242552e-01	[ 5.3901095e-01]	 1.9734836e-01


.. parsed-literal::

      24	 6.6030274e-01	 1.9343485e-01	 6.9696611e-01	 2.2047609e-01	[ 5.7625755e-01]	 2.0484757e-01


.. parsed-literal::

      25	 6.8270491e-01	 1.9149811e-01	 7.1970497e-01	 2.1908440e-01	[ 5.9647695e-01]	 2.0964909e-01
      26	 7.1044497e-01	 1.8847281e-01	 7.4710370e-01	 2.1631624e-01	[ 6.1850720e-01]	 1.9618464e-01


.. parsed-literal::

      27	 7.3590577e-01	 1.8546225e-01	 7.7082438e-01	 2.1197662e-01	[ 6.6427688e-01]	 1.9211650e-01


.. parsed-literal::

      28	 7.6621522e-01	 1.8540242e-01	 8.0178031e-01	 2.1314135e-01	[ 7.0876966e-01]	 2.0535064e-01


.. parsed-literal::

      29	 7.8976611e-01	 1.9024455e-01	 8.2689007e-01	 2.2201195e-01	[ 7.2781941e-01]	 2.0217204e-01
      30	 8.0387363e-01	 1.8786175e-01	 8.4271894e-01	 2.1868930e-01	[ 7.5442162e-01]	 2.0199800e-01


.. parsed-literal::

      31	 8.3113467e-01	 1.8983837e-01	 8.6945613e-01	 2.2106946e-01	[ 7.7842688e-01]	 2.0853710e-01


.. parsed-literal::

      32	 8.5553955e-01	 1.8500741e-01	 8.9375559e-01	 2.1536079e-01	[ 8.0445966e-01]	 2.0223236e-01


.. parsed-literal::

      33	 8.7681261e-01	 1.8137577e-01	 9.1575868e-01	 2.1072052e-01	[ 8.2521498e-01]	 2.0775700e-01


.. parsed-literal::

      34	 9.1653587e-01	 1.7392836e-01	 9.5720089e-01	 2.0009891e-01	[ 8.7472292e-01]	 2.0746255e-01


.. parsed-literal::

      35	 9.3780795e-01	 1.7189609e-01	 9.7868070e-01	 1.9773834e-01	[ 8.9708058e-01]	 2.0978069e-01


.. parsed-literal::

      36	 9.5247091e-01	 1.7058118e-01	 9.9369944e-01	 1.9598065e-01	[ 9.1394293e-01]	 2.1541858e-01
      37	 9.6673889e-01	 1.6952666e-01	 1.0084034e+00	 1.9560179e-01	[ 9.1826369e-01]	 2.0719910e-01


.. parsed-literal::

      38	 9.8201871e-01	 1.6949632e-01	 1.0244660e+00	 1.9672587e-01	[ 9.2450036e-01]	 1.9988775e-01
      39	 9.9716311e-01	 1.6920411e-01	 1.0400617e+00	 1.9643037e-01	[ 9.3520023e-01]	 1.9555950e-01


.. parsed-literal::

      40	 1.0085545e+00	 1.6825545e-01	 1.0517755e+00	 1.9565185e-01	[ 9.5078695e-01]	 2.1493578e-01


.. parsed-literal::

      41	 1.0192141e+00	 1.6680939e-01	 1.0619988e+00	 1.9463829e-01	[ 9.6956133e-01]	 2.0814490e-01


.. parsed-literal::

      42	 1.0298250e+00	 1.6480836e-01	 1.0728741e+00	 1.9270974e-01	[ 9.7914430e-01]	 2.0950079e-01


.. parsed-literal::

      43	 1.0458227e+00	 1.6128985e-01	 1.0899592e+00	 1.8887642e-01	[ 9.8836260e-01]	 2.0704794e-01


.. parsed-literal::

      44	 1.0579171e+00	 1.5933347e-01	 1.1030042e+00	 1.8548645e-01	[ 9.9381855e-01]	 2.1749973e-01


.. parsed-literal::

      45	 1.0689713e+00	 1.5854243e-01	 1.1140578e+00	 1.8450801e-01	[ 1.0053195e+00]	 2.1150923e-01
      46	 1.0758365e+00	 1.5821495e-01	 1.1208577e+00	 1.8429739e-01	[ 1.0116836e+00]	 1.9575644e-01


.. parsed-literal::

      47	 1.0834602e+00	 1.5737192e-01	 1.1285809e+00	 1.8379846e-01	[ 1.0165410e+00]	 1.7752051e-01


.. parsed-literal::

      48	 1.0945217e+00	 1.5613944e-01	 1.1401542e+00	 1.8359436e-01	  1.0096068e+00 	 2.1153498e-01


.. parsed-literal::

      49	 1.1031501e+00	 1.5471075e-01	 1.1490087e+00	 1.8204482e-01	[ 1.0168825e+00]	 2.0345092e-01
      50	 1.1100352e+00	 1.5358599e-01	 1.1560183e+00	 1.8062158e-01	[ 1.0229897e+00]	 1.9172120e-01


.. parsed-literal::

      51	 1.1182241e+00	 1.5199633e-01	 1.1645242e+00	 1.7902492e-01	[ 1.0243795e+00]	 2.0327830e-01


.. parsed-literal::

      52	 1.1257552e+00	 1.5043632e-01	 1.1724134e+00	 1.7691503e-01	[ 1.0281043e+00]	 2.1224213e-01


.. parsed-literal::

      53	 1.1338500e+00	 1.4974838e-01	 1.1805459e+00	 1.7644288e-01	[ 1.0336350e+00]	 2.1114492e-01


.. parsed-literal::

      54	 1.1414273e+00	 1.4866025e-01	 1.1884322e+00	 1.7562438e-01	  1.0285440e+00 	 2.0755267e-01
      55	 1.1495495e+00	 1.4792779e-01	 1.1967109e+00	 1.7435214e-01	[ 1.0361259e+00]	 1.6909337e-01


.. parsed-literal::

      56	 1.1588404e+00	 1.4717149e-01	 1.2062839e+00	 1.7369747e-01	[ 1.0365189e+00]	 2.1391320e-01
      57	 1.1679757e+00	 1.4668574e-01	 1.2159245e+00	 1.7312306e-01	[ 1.0441564e+00]	 2.0052242e-01


.. parsed-literal::

      58	 1.1768246e+00	 1.4598094e-01	 1.2248473e+00	 1.7291731e-01	[ 1.0493837e+00]	 2.1539617e-01
      59	 1.1842443e+00	 1.4534818e-01	 1.2323697e+00	 1.7261198e-01	[ 1.0523764e+00]	 2.0020604e-01


.. parsed-literal::

      60	 1.1935309e+00	 1.4429902e-01	 1.2419316e+00	 1.7223194e-01	[ 1.0606510e+00]	 1.9225645e-01
      61	 1.2001460e+00	 1.4401432e-01	 1.2490640e+00	 1.7282658e-01	  1.0557800e+00 	 1.9700599e-01


.. parsed-literal::

      62	 1.2082319e+00	 1.4381373e-01	 1.2570613e+00	 1.7278859e-01	[ 1.0713419e+00]	 2.0631647e-01
      63	 1.2140235e+00	 1.4368288e-01	 1.2628702e+00	 1.7308311e-01	[ 1.0786710e+00]	 1.9661403e-01


.. parsed-literal::

      64	 1.2217337e+00	 1.4397072e-01	 1.2709744e+00	 1.7394508e-01	  1.0779265e+00 	 1.9673347e-01
      65	 1.2293745e+00	 1.4373803e-01	 1.2789363e+00	 1.7487758e-01	[ 1.0859745e+00]	 1.8106246e-01


.. parsed-literal::

      66	 1.2354032e+00	 1.4371637e-01	 1.2851441e+00	 1.7523688e-01	  1.0835104e+00 	 2.0287490e-01


.. parsed-literal::

      67	 1.2417934e+00	 1.4289786e-01	 1.2916715e+00	 1.7434266e-01	[ 1.0925347e+00]	 2.0989275e-01
      68	 1.2485434e+00	 1.4191404e-01	 1.2988877e+00	 1.7411263e-01	  1.0870279e+00 	 1.9181991e-01


.. parsed-literal::

      69	 1.2546212e+00	 1.4110017e-01	 1.3051451e+00	 1.7315625e-01	[ 1.1053033e+00]	 1.9959569e-01


.. parsed-literal::

      70	 1.2592936e+00	 1.4080709e-01	 1.3097631e+00	 1.7313300e-01	[ 1.1092872e+00]	 2.1030974e-01


.. parsed-literal::

      71	 1.2645028e+00	 1.4079900e-01	 1.3149340e+00	 1.7343060e-01	[ 1.1156496e+00]	 2.0993495e-01
      72	 1.2714663e+00	 1.4034310e-01	 1.3221371e+00	 1.7307008e-01	[ 1.1210970e+00]	 2.0801473e-01


.. parsed-literal::

      73	 1.2782568e+00	 1.4037445e-01	 1.3291448e+00	 1.7299061e-01	[ 1.1328259e+00]	 2.1081758e-01


.. parsed-literal::

      74	 1.2836422e+00	 1.3950569e-01	 1.3346033e+00	 1.7212387e-01	[ 1.1370057e+00]	 2.0391226e-01


.. parsed-literal::

      75	 1.2894977e+00	 1.3862505e-01	 1.3406409e+00	 1.7095555e-01	[ 1.1427733e+00]	 2.0249891e-01
      76	 1.2946481e+00	 1.3838989e-01	 1.3460454e+00	 1.7063474e-01	[ 1.1434075e+00]	 1.9860148e-01


.. parsed-literal::

      77	 1.3002243e+00	 1.3831853e-01	 1.3517666e+00	 1.6989539e-01	[ 1.1455604e+00]	 1.9942856e-01


.. parsed-literal::

      78	 1.3054771e+00	 1.3839423e-01	 1.3569548e+00	 1.6997445e-01	[ 1.1478263e+00]	 2.1743870e-01
      79	 1.3125082e+00	 1.3847763e-01	 1.3639142e+00	 1.7025549e-01	[ 1.1491497e+00]	 1.8711042e-01


.. parsed-literal::

      80	 1.3187275e+00	 1.3797410e-01	 1.3704888e+00	 1.6894727e-01	[ 1.1507528e+00]	 2.0265079e-01


.. parsed-literal::

      81	 1.3255227e+00	 1.3738669e-01	 1.3774230e+00	 1.6829972e-01	[ 1.1533369e+00]	 2.1519995e-01
      82	 1.3308478e+00	 1.3673510e-01	 1.3829121e+00	 1.6748703e-01	  1.1532060e+00 	 1.9676995e-01


.. parsed-literal::

      83	 1.3361097e+00	 1.3624681e-01	 1.3885326e+00	 1.6639760e-01	[ 1.1553369e+00]	 1.9064736e-01


.. parsed-literal::

      84	 1.3416776e+00	 1.3574344e-01	 1.3942470e+00	 1.6547867e-01	  1.1546548e+00 	 2.1248293e-01


.. parsed-literal::

      85	 1.3474053e+00	 1.3546728e-01	 1.4000345e+00	 1.6481140e-01	[ 1.1557561e+00]	 2.0694041e-01


.. parsed-literal::

      86	 1.3530653e+00	 1.3546152e-01	 1.4058054e+00	 1.6457659e-01	[ 1.1580151e+00]	 2.0551348e-01


.. parsed-literal::

      87	 1.3578544e+00	 1.3501212e-01	 1.4105852e+00	 1.6379335e-01	  1.1578561e+00 	 2.0857167e-01


.. parsed-literal::

      88	 1.3621118e+00	 1.3488872e-01	 1.4149078e+00	 1.6336744e-01	[ 1.1599706e+00]	 2.5363517e-01


.. parsed-literal::

      89	 1.3653034e+00	 1.3442882e-01	 1.4181295e+00	 1.6266890e-01	[ 1.1604555e+00]	 2.1676064e-01


.. parsed-literal::

      90	 1.3678433e+00	 1.3422744e-01	 1.4205785e+00	 1.6243728e-01	[ 1.1661705e+00]	 2.0917964e-01


.. parsed-literal::

      91	 1.3720405e+00	 1.3420244e-01	 1.4247568e+00	 1.6245800e-01	[ 1.1697843e+00]	 2.0976114e-01
      92	 1.3746809e+00	 1.3423009e-01	 1.4276118e+00	 1.6174182e-01	[ 1.1734915e+00]	 1.9946313e-01


.. parsed-literal::

      93	 1.3789358e+00	 1.3437898e-01	 1.4316950e+00	 1.6231542e-01	  1.1731534e+00 	 2.0266509e-01


.. parsed-literal::

      94	 1.3813532e+00	 1.3449170e-01	 1.4341155e+00	 1.6254911e-01	  1.1705708e+00 	 2.0594954e-01
      95	 1.3844147e+00	 1.3461918e-01	 1.4372225e+00	 1.6267407e-01	  1.1689680e+00 	 1.9932818e-01


.. parsed-literal::

      96	 1.3892480e+00	 1.3466992e-01	 1.4420990e+00	 1.6247296e-01	  1.1709211e+00 	 2.0991826e-01


.. parsed-literal::

      97	 1.3919948e+00	 1.3484200e-01	 1.4449192e+00	 1.6240294e-01	[ 1.1758517e+00]	 3.2499528e-01


.. parsed-literal::

      98	 1.3949611e+00	 1.3457658e-01	 1.4478782e+00	 1.6180011e-01	[ 1.1809829e+00]	 2.1119714e-01


.. parsed-literal::

      99	 1.3973879e+00	 1.3429645e-01	 1.4503194e+00	 1.6122296e-01	[ 1.1858182e+00]	 2.0488143e-01
     100	 1.4004884e+00	 1.3399557e-01	 1.4534925e+00	 1.6063678e-01	[ 1.1886036e+00]	 2.0010781e-01


.. parsed-literal::

     101	 1.4032636e+00	 1.3403481e-01	 1.4564957e+00	 1.6028391e-01	[ 1.1958749e+00]	 2.1500850e-01
     102	 1.4076767e+00	 1.3366385e-01	 1.4608316e+00	 1.5978690e-01	  1.1957299e+00 	 1.9468760e-01


.. parsed-literal::

     103	 1.4103557e+00	 1.3355199e-01	 1.4634715e+00	 1.5970295e-01	  1.1958427e+00 	 2.0244241e-01


.. parsed-literal::

     104	 1.4141770e+00	 1.3332104e-01	 1.4673619e+00	 1.5918594e-01	  1.1955866e+00 	 2.0728350e-01


.. parsed-literal::

     105	 1.4155740e+00	 1.3307866e-01	 1.4690212e+00	 1.5828740e-01	[ 1.1967992e+00]	 2.1649384e-01


.. parsed-literal::

     106	 1.4199261e+00	 1.3293347e-01	 1.4733119e+00	 1.5782338e-01	[ 1.1986513e+00]	 2.1729565e-01


.. parsed-literal::

     107	 1.4218330e+00	 1.3280240e-01	 1.4751916e+00	 1.5752571e-01	[ 1.1998135e+00]	 2.1267676e-01


.. parsed-literal::

     108	 1.4243316e+00	 1.3268656e-01	 1.4777209e+00	 1.5714972e-01	[ 1.2000331e+00]	 2.1717453e-01


.. parsed-literal::

     109	 1.4283205e+00	 1.3278571e-01	 1.4817207e+00	 1.5724582e-01	  1.1968588e+00 	 2.0089221e-01


.. parsed-literal::

     110	 1.4307404e+00	 1.3301291e-01	 1.4842278e+00	 1.5774266e-01	  1.1936429e+00 	 3.2175612e-01
     111	 1.4333240e+00	 1.3312372e-01	 1.4868120e+00	 1.5793634e-01	  1.1905371e+00 	 2.0278144e-01


.. parsed-literal::

     112	 1.4358090e+00	 1.3325882e-01	 1.4893039e+00	 1.5821000e-01	  1.1889914e+00 	 2.1276617e-01


.. parsed-literal::

     113	 1.4387745e+00	 1.3327275e-01	 1.4923925e+00	 1.5841551e-01	  1.1809603e+00 	 2.1205640e-01


.. parsed-literal::

     114	 1.4417740e+00	 1.3346851e-01	 1.4954379e+00	 1.5875024e-01	  1.1819626e+00 	 2.0233369e-01
     115	 1.4441258e+00	 1.3335673e-01	 1.4978010e+00	 1.5868721e-01	  1.1829560e+00 	 1.9092083e-01


.. parsed-literal::

     116	 1.4478843e+00	 1.3325342e-01	 1.5016895e+00	 1.5869084e-01	  1.1809611e+00 	 2.1271658e-01


.. parsed-literal::

     117	 1.4497964e+00	 1.3329007e-01	 1.5036816e+00	 1.5907188e-01	  1.1735509e+00 	 3.2569814e-01


.. parsed-literal::

     118	 1.4521075e+00	 1.3326639e-01	 1.5060351e+00	 1.5917460e-01	  1.1709181e+00 	 2.1605945e-01


.. parsed-literal::

     119	 1.4548460e+00	 1.3337315e-01	 1.5088413e+00	 1.5951583e-01	  1.1642742e+00 	 2.0932078e-01


.. parsed-literal::

     120	 1.4572906e+00	 1.3333815e-01	 1.5113218e+00	 1.5952907e-01	  1.1646245e+00 	 2.0957112e-01


.. parsed-literal::

     121	 1.4600690e+00	 1.3344830e-01	 1.5141302e+00	 1.5971474e-01	  1.1631574e+00 	 2.1219087e-01


.. parsed-literal::

     122	 1.4624462e+00	 1.3343669e-01	 1.5165812e+00	 1.5961661e-01	  1.1634794e+00 	 2.1378589e-01


.. parsed-literal::

     123	 1.4645830e+00	 1.3340247e-01	 1.5187983e+00	 1.5930582e-01	  1.1654623e+00 	 2.0980239e-01


.. parsed-literal::

     124	 1.4664192e+00	 1.3325760e-01	 1.5207491e+00	 1.5918905e-01	  1.1571584e+00 	 2.0904541e-01


.. parsed-literal::

     125	 1.4680731e+00	 1.3315860e-01	 1.5223857e+00	 1.5893797e-01	  1.1577383e+00 	 2.0609522e-01


.. parsed-literal::

     126	 1.4701646e+00	 1.3294977e-01	 1.5244923e+00	 1.5855215e-01	  1.1535746e+00 	 2.1457863e-01


.. parsed-literal::

     127	 1.4724017e+00	 1.3283986e-01	 1.5268247e+00	 1.5827717e-01	  1.1485999e+00 	 2.1095848e-01


.. parsed-literal::

     128	 1.4747636e+00	 1.3255654e-01	 1.5292704e+00	 1.5771480e-01	  1.1419073e+00 	 2.0232511e-01


.. parsed-literal::

     129	 1.4762326e+00	 1.3258862e-01	 1.5307830e+00	 1.5783395e-01	  1.1393276e+00 	 2.0876670e-01


.. parsed-literal::

     130	 1.4775790e+00	 1.3265203e-01	 1.5320754e+00	 1.5795173e-01	  1.1442859e+00 	 2.2123027e-01


.. parsed-literal::

     131	 1.4801251e+00	 1.3275382e-01	 1.5346188e+00	 1.5806417e-01	  1.1491103e+00 	 2.2044277e-01
     132	 1.4817349e+00	 1.3293886e-01	 1.5363169e+00	 1.5825921e-01	  1.1497987e+00 	 1.9775295e-01


.. parsed-literal::

     133	 1.4834109e+00	 1.3293366e-01	 1.5379924e+00	 1.5819091e-01	  1.1482036e+00 	 1.9110680e-01
     134	 1.4858305e+00	 1.3293957e-01	 1.5405139e+00	 1.5802703e-01	  1.1401081e+00 	 1.9735718e-01


.. parsed-literal::

     135	 1.4872150e+00	 1.3296179e-01	 1.5419267e+00	 1.5801771e-01	  1.1375238e+00 	 2.0088458e-01


.. parsed-literal::

     136	 1.4884193e+00	 1.3302482e-01	 1.5433271e+00	 1.5822791e-01	  1.1193869e+00 	 2.0812631e-01


.. parsed-literal::

     137	 1.4907337e+00	 1.3303044e-01	 1.5455088e+00	 1.5817569e-01	  1.1294622e+00 	 2.0732594e-01
     138	 1.4914849e+00	 1.3302335e-01	 1.5462364e+00	 1.5815386e-01	  1.1313644e+00 	 1.8991470e-01


.. parsed-literal::

     139	 1.4937617e+00	 1.3314422e-01	 1.5485529e+00	 1.5811016e-01	  1.1331584e+00 	 2.1722865e-01
     140	 1.4948039e+00	 1.3322433e-01	 1.5497266e+00	 1.5807767e-01	  1.1235906e+00 	 2.0378232e-01


.. parsed-literal::

     141	 1.4964382e+00	 1.3324360e-01	 1.5513222e+00	 1.5802002e-01	  1.1252608e+00 	 2.0412493e-01


.. parsed-literal::

     142	 1.4974694e+00	 1.3330152e-01	 1.5523962e+00	 1.5797700e-01	  1.1228680e+00 	 2.0723605e-01


.. parsed-literal::

     143	 1.4986764e+00	 1.3336778e-01	 1.5536575e+00	 1.5792524e-01	  1.1193701e+00 	 2.1260095e-01


.. parsed-literal::

     144	 1.5008276e+00	 1.3351115e-01	 1.5559094e+00	 1.5784013e-01	  1.1124853e+00 	 2.0378375e-01


.. parsed-literal::

     145	 1.5019420e+00	 1.3358285e-01	 1.5570888e+00	 1.5786239e-01	  1.1062018e+00 	 3.2718945e-01


.. parsed-literal::

     146	 1.5033453e+00	 1.3361368e-01	 1.5584921e+00	 1.5785666e-01	  1.1052596e+00 	 2.1433949e-01


.. parsed-literal::

     147	 1.5045261e+00	 1.3360909e-01	 1.5596694e+00	 1.5784897e-01	  1.1045999e+00 	 2.1317267e-01


.. parsed-literal::

     148	 1.5057418e+00	 1.3364097e-01	 1.5609224e+00	 1.5801424e-01	  1.1025881e+00 	 2.1873832e-01
     149	 1.5072406e+00	 1.3367628e-01	 1.5624579e+00	 1.5809467e-01	  1.0990307e+00 	 1.8597269e-01


.. parsed-literal::

     150	 1.5087441e+00	 1.3370921e-01	 1.5640181e+00	 1.5824648e-01	  1.0947616e+00 	 2.1730590e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.11 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd280d20b80>



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
    CPU times: user 2.03 s, sys: 48.9 ms, total: 2.08 s
    Wall time: 624 ms


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

