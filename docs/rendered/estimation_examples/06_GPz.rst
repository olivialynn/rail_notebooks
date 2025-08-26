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
       1	-3.3507076e-01	 3.1828969e-01	-3.2528531e-01	 3.2917929e-01	[-3.4685407e-01]	 4.6546268e-01


.. parsed-literal::

       2	-2.6305510e-01	 3.0653588e-01	-2.3755455e-01	 3.1716217e-01	[-2.7148069e-01]	 2.2850275e-01


.. parsed-literal::

       3	-2.2015960e-01	 2.8679232e-01	-1.7685585e-01	 2.9736872e-01	[-2.2276401e-01]	 2.8692627e-01
       4	-1.9236876e-01	 2.6358241e-01	-1.5149540e-01	 2.7276259e-01	[-2.0711939e-01]	 1.9685149e-01


.. parsed-literal::

       5	-1.0259138e-01	 2.5626244e-01	-6.5862769e-02	 2.6249057e-01	[-9.6024731e-02]	 2.1791935e-01


.. parsed-literal::

       6	-6.7979653e-02	 2.5048768e-01	-3.5482506e-02	 2.5617436e-01	[-5.7571744e-02]	 2.0725465e-01


.. parsed-literal::

       7	-4.9593019e-02	 2.4781298e-01	-2.4199181e-02	 2.5333315e-01	[-4.6634566e-02]	 2.1009445e-01
       8	-3.7731548e-02	 2.4596544e-01	-1.6396923e-02	 2.5162345e-01	[-4.0009513e-02]	 1.8739152e-01


.. parsed-literal::

       9	-2.3090952e-02	 2.4330691e-01	-5.0275434e-03	 2.4922404e-01	[-2.9874879e-02]	 2.1514297e-01
      10	-1.0748235e-02	 2.4074883e-01	 4.8872570e-03	 2.4724213e-01	[-2.5130238e-02]	 1.8898845e-01


.. parsed-literal::

      11	-5.3341703e-03	 2.4003223e-01	 8.8287902e-03	 2.4686445e-01	[-1.8944393e-02]	 1.8210173e-01
      12	-1.5441830e-03	 2.3944306e-01	 1.2515890e-02	 2.4656798e-01	[-1.7843193e-02]	 1.8312788e-01


.. parsed-literal::

      13	 1.5529211e-03	 2.3883901e-01	 1.5573775e-02	 2.4632934e-01	[-1.7566907e-02]	 2.1513796e-01
      14	 5.6072474e-03	 2.3799364e-01	 2.0095667e-02	 2.4592461e-01	[-1.5314992e-02]	 1.9830418e-01


.. parsed-literal::

      15	 1.5281905e-01	 2.2104789e-01	 1.7636255e-01	 2.3011926e-01	[ 1.4872131e-01]	 3.3268619e-01
      16	 1.6195613e-01	 2.1799176e-01	 1.8681690e-01	 2.2390200e-01	[ 1.7949942e-01]	 1.8858743e-01


.. parsed-literal::

      17	 2.8003381e-01	 2.0987174e-01	 3.1035257e-01	 2.1960220e-01	[ 2.9110487e-01]	 2.1359563e-01


.. parsed-literal::

      18	 3.3319237e-01	 2.0704493e-01	 3.6637742e-01	 2.1644386e-01	[ 3.3859533e-01]	 3.0345368e-01


.. parsed-literal::

      19	 3.6258679e-01	 2.0312224e-01	 3.9793890e-01	 2.1424540e-01	[ 3.6291406e-01]	 2.1631336e-01


.. parsed-literal::

      20	 3.8617046e-01	 2.0086227e-01	 4.2077089e-01	 2.1242371e-01	[ 3.9085295e-01]	 2.1011734e-01
      21	 4.2951600e-01	 1.9684649e-01	 4.6326989e-01	 2.0847440e-01	[ 4.3766111e-01]	 1.7409492e-01


.. parsed-literal::

      22	 5.0085161e-01	 1.9301747e-01	 5.3485848e-01	 2.0350763e-01	[ 5.2234055e-01]	 2.0257139e-01
      23	 5.9932961e-01	 1.8798877e-01	 6.3705576e-01	 1.9750496e-01	[ 6.4594552e-01]	 1.8588805e-01


.. parsed-literal::

      24	 6.1645892e-01	 1.8542682e-01	 6.5806183e-01	 1.9288739e-01	[ 6.7158273e-01]	 2.0540619e-01


.. parsed-literal::

      25	 6.8127347e-01	 1.8148307e-01	 7.1964646e-01	 1.9125115e-01	[ 7.2779518e-01]	 2.0955110e-01


.. parsed-literal::

      26	 7.0201372e-01	 1.7938555e-01	 7.4010414e-01	 1.9102189e-01	[ 7.4709360e-01]	 2.1026921e-01


.. parsed-literal::

      27	 7.3859550e-01	 1.7604822e-01	 7.7623610e-01	 1.9006401e-01	[ 7.8430661e-01]	 2.1255803e-01


.. parsed-literal::

      28	 7.5218094e-01	 1.8327589e-01	 7.9088548e-01	 1.9743332e-01	[ 8.0388345e-01]	 2.0711112e-01


.. parsed-literal::

      29	 7.8123381e-01	 1.7692195e-01	 8.1994663e-01	 1.9253068e-01	[ 8.2423139e-01]	 2.0408058e-01
      30	 7.9798576e-01	 1.7436399e-01	 8.3673466e-01	 1.9029899e-01	[ 8.3748350e-01]	 2.0418072e-01


.. parsed-literal::

      31	 8.2324696e-01	 1.7134659e-01	 8.6240289e-01	 1.8731861e-01	[ 8.5866313e-01]	 2.0266366e-01
      32	 8.3867793e-01	 1.7051525e-01	 8.7758124e-01	 1.8843157e-01	[ 8.7536248e-01]	 1.9148135e-01


.. parsed-literal::

      33	 8.6037321e-01	 1.6979394e-01	 9.0047529e-01	 1.8542887e-01	[ 8.9398516e-01]	 2.0451760e-01
      34	 8.8425217e-01	 1.6699135e-01	 9.2360330e-01	 1.8387395e-01	[ 9.1590283e-01]	 1.8419719e-01


.. parsed-literal::

      35	 9.0387739e-01	 1.6382457e-01	 9.4313737e-01	 1.8127433e-01	[ 9.3189748e-01]	 2.0863152e-01


.. parsed-literal::

      36	 9.3366962e-01	 1.6110210e-01	 9.7377911e-01	 1.8036351e-01	[ 9.5456234e-01]	 2.1466994e-01
      37	 9.4486750e-01	 1.5919497e-01	 9.8770963e-01	 1.7669976e-01	  9.4928383e-01 	 1.7914152e-01


.. parsed-literal::

      38	 9.6800399e-01	 1.5595260e-01	 1.0108408e+00	 1.7445664e-01	[ 9.6896970e-01]	 1.9551301e-01


.. parsed-literal::

      39	 9.7958780e-01	 1.5481017e-01	 1.0220177e+00	 1.7418653e-01	[ 9.8167939e-01]	 2.1461821e-01


.. parsed-literal::

      40	 9.9698938e-01	 1.5271199e-01	 1.0397329e+00	 1.7293186e-01	[ 9.9366177e-01]	 2.0897484e-01
      41	 1.0147312e+00	 1.5112300e-01	 1.0580336e+00	 1.7191406e-01	[ 1.0024783e+00]	 1.8507361e-01


.. parsed-literal::

      42	 1.0295166e+00	 1.4932159e-01	 1.0740847e+00	 1.6851917e-01	[ 1.0100341e+00]	 1.7375398e-01


.. parsed-literal::

      43	 1.0425790e+00	 1.4917110e-01	 1.0872748e+00	 1.6821047e-01	[ 1.0220210e+00]	 2.0991325e-01


.. parsed-literal::

      44	 1.0486578e+00	 1.4903630e-01	 1.0933119e+00	 1.6862342e-01	[ 1.0249258e+00]	 2.0599580e-01
      45	 1.0660044e+00	 1.4877554e-01	 1.1112163e+00	 1.7063087e-01	[ 1.0278945e+00]	 1.8448567e-01


.. parsed-literal::

      46	 1.0779328e+00	 1.4848670e-01	 1.1241853e+00	 1.7270909e-01	  1.0198763e+00 	 2.0970988e-01


.. parsed-literal::

      47	 1.0906844e+00	 1.4746039e-01	 1.1370866e+00	 1.7233178e-01	[ 1.0336119e+00]	 2.1435785e-01


.. parsed-literal::

      48	 1.0971426e+00	 1.4659688e-01	 1.1435030e+00	 1.7108892e-01	[ 1.0412552e+00]	 2.1285605e-01
      49	 1.1088583e+00	 1.4490219e-01	 1.1555111e+00	 1.6810480e-01	[ 1.0559519e+00]	 1.9862294e-01


.. parsed-literal::

      50	 1.1205029e+00	 1.4316670e-01	 1.1672135e+00	 1.6429552e-01	[ 1.0736094e+00]	 2.1503258e-01


.. parsed-literal::

      51	 1.1320673e+00	 1.4198639e-01	 1.1788741e+00	 1.6208345e-01	[ 1.0872028e+00]	 2.1100426e-01
      52	 1.1423646e+00	 1.4076338e-01	 1.1893994e+00	 1.6015365e-01	[ 1.0981023e+00]	 1.8884087e-01


.. parsed-literal::

      53	 1.1530234e+00	 1.3966131e-01	 1.2003658e+00	 1.5815898e-01	[ 1.1068193e+00]	 1.9373131e-01


.. parsed-literal::

      54	 1.1624187e+00	 1.3799157e-01	 1.2102254e+00	 1.5595537e-01	[ 1.1087716e+00]	 2.1684217e-01


.. parsed-literal::

      55	 1.1720277e+00	 1.3767824e-01	 1.2196600e+00	 1.5558468e-01	[ 1.1162397e+00]	 2.1127892e-01


.. parsed-literal::

      56	 1.1779769e+00	 1.3723241e-01	 1.2256739e+00	 1.5538178e-01	[ 1.1182915e+00]	 2.0719624e-01


.. parsed-literal::

      57	 1.1890907e+00	 1.3636691e-01	 1.2371659e+00	 1.5555715e-01	[ 1.1198303e+00]	 2.1297979e-01
      58	 1.1964637e+00	 1.3605196e-01	 1.2449505e+00	 1.5818641e-01	  1.1094151e+00 	 1.8083787e-01


.. parsed-literal::

      59	 1.2062468e+00	 1.3570677e-01	 1.2547412e+00	 1.5776319e-01	[ 1.1211505e+00]	 1.9971323e-01


.. parsed-literal::

      60	 1.2115511e+00	 1.3565480e-01	 1.2601053e+00	 1.5798987e-01	[ 1.1243671e+00]	 2.0484066e-01


.. parsed-literal::

      61	 1.2168472e+00	 1.3557327e-01	 1.2656939e+00	 1.5841419e-01	  1.1240853e+00 	 2.0964360e-01
      62	 1.2256464e+00	 1.3560543e-01	 1.2748436e+00	 1.5928855e-01	  1.1217581e+00 	 1.9698954e-01


.. parsed-literal::

      63	 1.2336726e+00	 1.3508416e-01	 1.2832205e+00	 1.5884140e-01	  1.1188984e+00 	 2.0313263e-01


.. parsed-literal::

      64	 1.2400841e+00	 1.3483304e-01	 1.2895777e+00	 1.5830698e-01	[ 1.1277246e+00]	 3.0801797e-01


.. parsed-literal::

      65	 1.2471103e+00	 1.3413715e-01	 1.2966672e+00	 1.5701841e-01	[ 1.1369424e+00]	 2.1183681e-01
      66	 1.2566135e+00	 1.3278982e-01	 1.3063949e+00	 1.5422223e-01	[ 1.1510353e+00]	 1.9978952e-01


.. parsed-literal::

      67	 1.2645140e+00	 1.3171000e-01	 1.3146646e+00	 1.5233130e-01	[ 1.1535084e+00]	 3.1094146e-01


.. parsed-literal::

      68	 1.2729354e+00	 1.3060996e-01	 1.3233272e+00	 1.4948766e-01	[ 1.1620803e+00]	 2.0710731e-01


.. parsed-literal::

      69	 1.2791296e+00	 1.3010076e-01	 1.3296564e+00	 1.4878982e-01	[ 1.1682679e+00]	 2.0644712e-01


.. parsed-literal::

      70	 1.2870548e+00	 1.2975881e-01	 1.3378442e+00	 1.4810178e-01	[ 1.1724215e+00]	 2.1452284e-01


.. parsed-literal::

      71	 1.2953452e+00	 1.2938360e-01	 1.3463529e+00	 1.4722376e-01	[ 1.1839020e+00]	 2.0998931e-01


.. parsed-literal::

      72	 1.3043545e+00	 1.2935407e-01	 1.3554094e+00	 1.4614394e-01	[ 1.1972277e+00]	 2.0903969e-01
      73	 1.3124930e+00	 1.2922070e-01	 1.3637801e+00	 1.4510971e-01	[ 1.2094445e+00]	 1.9661951e-01


.. parsed-literal::

      74	 1.3190405e+00	 1.2915742e-01	 1.3704566e+00	 1.4402176e-01	[ 1.2187060e+00]	 1.9551301e-01


.. parsed-literal::

      75	 1.3267899e+00	 1.2866723e-01	 1.3781986e+00	 1.4386545e-01	[ 1.2232889e+00]	 2.1108389e-01
      76	 1.3334873e+00	 1.2803375e-01	 1.3850304e+00	 1.4395817e-01	  1.2224064e+00 	 1.9012976e-01


.. parsed-literal::

      77	 1.3412097e+00	 1.2739755e-01	 1.3930666e+00	 1.4381095e-01	  1.2214967e+00 	 2.0731544e-01
      78	 1.3449164e+00	 1.2760528e-01	 1.3972043e+00	 1.4526316e-01	  1.2049689e+00 	 1.9186592e-01


.. parsed-literal::

      79	 1.3530954e+00	 1.2702474e-01	 1.4049737e+00	 1.4432158e-01	  1.2170644e+00 	 2.0432568e-01


.. parsed-literal::

      80	 1.3559340e+00	 1.2692970e-01	 1.4077196e+00	 1.4393474e-01	  1.2214688e+00 	 2.0645452e-01


.. parsed-literal::

      81	 1.3625348e+00	 1.2657978e-01	 1.4143800e+00	 1.4333258e-01	  1.2196608e+00 	 2.0572543e-01


.. parsed-literal::

      82	 1.3692540e+00	 1.2647005e-01	 1.4213069e+00	 1.4285912e-01	  1.2214641e+00 	 2.0402455e-01
      83	 1.3753178e+00	 1.2600947e-01	 1.4273640e+00	 1.4241154e-01	  1.2196330e+00 	 1.8298531e-01


.. parsed-literal::

      84	 1.3802884e+00	 1.2553857e-01	 1.4325268e+00	 1.4218078e-01	  1.2167726e+00 	 2.0104694e-01


.. parsed-literal::

      85	 1.3857856e+00	 1.2492481e-01	 1.4383470e+00	 1.4170922e-01	  1.2164556e+00 	 2.0366025e-01
      86	 1.3925191e+00	 1.2407247e-01	 1.4454712e+00	 1.4077240e-01	  1.2213229e+00 	 1.9741440e-01


.. parsed-literal::

      87	 1.3978427e+00	 1.2383066e-01	 1.4507327e+00	 1.4054397e-01	[ 1.2313410e+00]	 1.8718600e-01


.. parsed-literal::

      88	 1.4019656e+00	 1.2356559e-01	 1.4548420e+00	 1.4008584e-01	[ 1.2403060e+00]	 2.0581055e-01
      89	 1.4061143e+00	 1.2371586e-01	 1.4591783e+00	 1.4009946e-01	[ 1.2427425e+00]	 1.7431760e-01


.. parsed-literal::

      90	 1.4107446e+00	 1.2363567e-01	 1.4638649e+00	 1.3985762e-01	[ 1.2487773e+00]	 2.1082377e-01
      91	 1.4151371e+00	 1.2350207e-01	 1.4683947e+00	 1.3989357e-01	[ 1.2495997e+00]	 1.8329978e-01


.. parsed-literal::

      92	 1.4193225e+00	 1.2351770e-01	 1.4726964e+00	 1.3993845e-01	  1.2474219e+00 	 2.1757889e-01
      93	 1.4231963e+00	 1.2313977e-01	 1.4767056e+00	 1.4017494e-01	  1.2391783e+00 	 1.9420481e-01


.. parsed-literal::

      94	 1.4269673e+00	 1.2291183e-01	 1.4803695e+00	 1.3951454e-01	  1.2430273e+00 	 1.9972253e-01


.. parsed-literal::

      95	 1.4293390e+00	 1.2273353e-01	 1.4827349e+00	 1.3920508e-01	  1.2458187e+00 	 2.1120977e-01


.. parsed-literal::

      96	 1.4327676e+00	 1.2223612e-01	 1.4861390e+00	 1.3871087e-01	  1.2490583e+00 	 2.0848560e-01


.. parsed-literal::

      97	 1.4354977e+00	 1.2216886e-01	 1.4890763e+00	 1.3938521e-01	[ 1.2557171e+00]	 2.1171570e-01


.. parsed-literal::

      98	 1.4388097e+00	 1.2191865e-01	 1.4923285e+00	 1.3918214e-01	[ 1.2570230e+00]	 2.0929480e-01


.. parsed-literal::

      99	 1.4413746e+00	 1.2190969e-01	 1.4949261e+00	 1.3925951e-01	[ 1.2588854e+00]	 2.2042108e-01


.. parsed-literal::

     100	 1.4443383e+00	 1.2190852e-01	 1.4979253e+00	 1.3950234e-01	[ 1.2615831e+00]	 2.0421243e-01


.. parsed-literal::

     101	 1.4465375e+00	 1.2220802e-01	 1.5002938e+00	 1.3989986e-01	[ 1.2727922e+00]	 2.1536875e-01


.. parsed-literal::

     102	 1.4507076e+00	 1.2212257e-01	 1.5043237e+00	 1.3985101e-01	[ 1.2745206e+00]	 2.1490622e-01
     103	 1.4525342e+00	 1.2204699e-01	 1.5061226e+00	 1.3976463e-01	[ 1.2755986e+00]	 1.9822383e-01


.. parsed-literal::

     104	 1.4546880e+00	 1.2198136e-01	 1.5082888e+00	 1.3957056e-01	[ 1.2773543e+00]	 2.0660710e-01


.. parsed-literal::

     105	 1.4573255e+00	 1.2183567e-01	 1.5110485e+00	 1.3982587e-01	  1.2741270e+00 	 2.1674418e-01


.. parsed-literal::

     106	 1.4605394e+00	 1.2175957e-01	 1.5143141e+00	 1.3932896e-01	[ 1.2776962e+00]	 2.0889211e-01


.. parsed-literal::

     107	 1.4628852e+00	 1.2173203e-01	 1.5167260e+00	 1.3922813e-01	[ 1.2782308e+00]	 2.0095873e-01


.. parsed-literal::

     108	 1.4653205e+00	 1.2166098e-01	 1.5192303e+00	 1.3941018e-01	  1.2757412e+00 	 2.0224857e-01


.. parsed-literal::

     109	 1.4674779e+00	 1.2160613e-01	 1.5214478e+00	 1.3951291e-01	  1.2723055e+00 	 2.1768045e-01


.. parsed-literal::

     110	 1.4699732e+00	 1.2145434e-01	 1.5239371e+00	 1.3991383e-01	  1.2702801e+00 	 2.1076345e-01


.. parsed-literal::

     111	 1.4717817e+00	 1.2134770e-01	 1.5257048e+00	 1.4011256e-01	  1.2687794e+00 	 2.0666265e-01
     112	 1.4743141e+00	 1.2118987e-01	 1.5282317e+00	 1.4025798e-01	  1.2672722e+00 	 1.8501091e-01


.. parsed-literal::

     113	 1.4770607e+00	 1.2101120e-01	 1.5310460e+00	 1.4029991e-01	  1.2568570e+00 	 2.0411372e-01


.. parsed-literal::

     114	 1.4802545e+00	 1.2086056e-01	 1.5342604e+00	 1.4012206e-01	  1.2568749e+00 	 2.0161319e-01
     115	 1.4828872e+00	 1.2080949e-01	 1.5369253e+00	 1.3983071e-01	  1.2579546e+00 	 1.6317725e-01


.. parsed-literal::

     116	 1.4855162e+00	 1.2074689e-01	 1.5396581e+00	 1.3960118e-01	  1.2554606e+00 	 2.0417261e-01


.. parsed-literal::

     117	 1.4883564e+00	 1.2060406e-01	 1.5426558e+00	 1.3955607e-01	  1.2517101e+00 	 2.0809889e-01


.. parsed-literal::

     118	 1.4911019e+00	 1.2035069e-01	 1.5454473e+00	 1.3954296e-01	  1.2468077e+00 	 2.1082973e-01


.. parsed-literal::

     119	 1.4933836e+00	 1.2007988e-01	 1.5478263e+00	 1.3967540e-01	  1.2347602e+00 	 2.2223639e-01
     120	 1.4950173e+00	 1.1992077e-01	 1.5494050e+00	 1.3986986e-01	  1.2325467e+00 	 1.7837501e-01


.. parsed-literal::

     121	 1.4960990e+00	 1.1987319e-01	 1.5504662e+00	 1.3988491e-01	  1.2322931e+00 	 1.9538498e-01
     122	 1.4988298e+00	 1.1973400e-01	 1.5532597e+00	 1.3994079e-01	  1.2236507e+00 	 1.8068194e-01


.. parsed-literal::

     123	 1.4998702e+00	 1.1968140e-01	 1.5544059e+00	 1.4020739e-01	  1.2066313e+00 	 2.0080853e-01


.. parsed-literal::

     124	 1.5019347e+00	 1.1960545e-01	 1.5564273e+00	 1.4000301e-01	  1.2104883e+00 	 2.1340632e-01


.. parsed-literal::

     125	 1.5033472e+00	 1.1954369e-01	 1.5578549e+00	 1.3994162e-01	  1.2103994e+00 	 2.0423198e-01


.. parsed-literal::

     126	 1.5048483e+00	 1.1947889e-01	 1.5593790e+00	 1.3994747e-01	  1.2091547e+00 	 2.1143031e-01


.. parsed-literal::

     127	 1.5073434e+00	 1.1938413e-01	 1.5619308e+00	 1.3985192e-01	  1.2054550e+00 	 2.1044326e-01


.. parsed-literal::

     128	 1.5089030e+00	 1.1939434e-01	 1.5635606e+00	 1.4015303e-01	  1.2009216e+00 	 3.2201457e-01
     129	 1.5114411e+00	 1.1932341e-01	 1.5661503e+00	 1.3988387e-01	  1.1943609e+00 	 1.9799876e-01


.. parsed-literal::

     130	 1.5132940e+00	 1.1932774e-01	 1.5680198e+00	 1.3976488e-01	  1.1925618e+00 	 1.9963765e-01


.. parsed-literal::

     131	 1.5152417e+00	 1.1936033e-01	 1.5700362e+00	 1.3956293e-01	  1.1859978e+00 	 2.0946002e-01


.. parsed-literal::

     132	 1.5165955e+00	 1.1945362e-01	 1.5714317e+00	 1.3969676e-01	  1.1802893e+00 	 2.0862293e-01
     133	 1.5178573e+00	 1.1939712e-01	 1.5726488e+00	 1.3971843e-01	  1.1822878e+00 	 1.9735122e-01


.. parsed-literal::

     134	 1.5193295e+00	 1.1929339e-01	 1.5740881e+00	 1.3964400e-01	  1.1848267e+00 	 2.0681739e-01


.. parsed-literal::

     135	 1.5205459e+00	 1.1920701e-01	 1.5753066e+00	 1.3964238e-01	  1.1850150e+00 	 2.0419836e-01
     136	 1.5219520e+00	 1.1911756e-01	 1.5768641e+00	 1.3935163e-01	  1.1863624e+00 	 2.0614982e-01


.. parsed-literal::

     137	 1.5243076e+00	 1.1898295e-01	 1.5791498e+00	 1.3958090e-01	  1.1828426e+00 	 1.9842863e-01


.. parsed-literal::

     138	 1.5251036e+00	 1.1893366e-01	 1.5799255e+00	 1.3958750e-01	  1.1818415e+00 	 2.0150757e-01


.. parsed-literal::

     139	 1.5266534e+00	 1.1888080e-01	 1.5815211e+00	 1.3954964e-01	  1.1813189e+00 	 2.0481372e-01
     140	 1.5279050e+00	 1.1885403e-01	 1.5828616e+00	 1.3972307e-01	  1.1742702e+00 	 1.7712498e-01


.. parsed-literal::

     141	 1.5296682e+00	 1.1886489e-01	 1.5845724e+00	 1.3959755e-01	  1.1808130e+00 	 2.1626854e-01
     142	 1.5311596e+00	 1.1892997e-01	 1.5860366e+00	 1.3954976e-01	  1.1881871e+00 	 2.0113444e-01


.. parsed-literal::

     143	 1.5319904e+00	 1.1896964e-01	 1.5868460e+00	 1.3954544e-01	  1.1910684e+00 	 1.9613147e-01
     144	 1.5346707e+00	 1.1900551e-01	 1.5894970e+00	 1.3976188e-01	  1.1989228e+00 	 1.7989421e-01


.. parsed-literal::

     145	 1.5358123e+00	 1.1908147e-01	 1.5906496e+00	 1.3983668e-01	  1.1977753e+00 	 3.2412696e-01
     146	 1.5371698e+00	 1.1900342e-01	 1.5920117e+00	 1.3995968e-01	  1.1957988e+00 	 1.8361473e-01


.. parsed-literal::

     147	 1.5383690e+00	 1.1888352e-01	 1.5932568e+00	 1.4018274e-01	  1.1908729e+00 	 2.0782661e-01


.. parsed-literal::

     148	 1.5395871e+00	 1.1873054e-01	 1.5946008e+00	 1.4023761e-01	  1.1803756e+00 	 2.1825767e-01
     149	 1.5409101e+00	 1.1867705e-01	 1.5959555e+00	 1.4044506e-01	  1.1756289e+00 	 1.8090129e-01


.. parsed-literal::

     150	 1.5427377e+00	 1.1862540e-01	 1.5978639e+00	 1.4069603e-01	  1.1694610e+00 	 2.0300341e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.05 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd50cd02740>



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

.. parsed-literal::

    
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449


.. parsed-literal::

    CPU times: user 2.06 s, sys: 53 ms, total: 2.11 s
    Wall time: 635 ms


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

