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
       1	-3.4520701e-01	 3.2137564e-01	-3.3553517e-01	 3.1740177e-01	[-3.2838112e-01]	 4.6913218e-01


.. parsed-literal::

       2	-2.7655108e-01	 3.1143134e-01	-2.5348196e-01	 3.0734830e-01	[-2.4136669e-01]	 2.3365188e-01


.. parsed-literal::

       3	-2.3335379e-01	 2.9109560e-01	-1.9246044e-01	 2.8325775e-01	[-1.6326868e-01]	 2.8270626e-01


.. parsed-literal::

       4	-1.9287637e-01	 2.6741769e-01	-1.5159333e-01	 2.6012348e-01	[-1.1005778e-01]	 2.0991850e-01


.. parsed-literal::

       5	-1.0769346e-01	 2.5792360e-01	-7.5739533e-02	 2.5143398e-01	[-4.4651166e-02]	 2.1527028e-01


.. parsed-literal::

       6	-7.4061940e-02	 2.5280926e-01	-4.5206443e-02	 2.4692696e-01	[-2.2081695e-02]	 2.0975423e-01


.. parsed-literal::

       7	-5.7391355e-02	 2.5024088e-01	-3.4043140e-02	 2.4417057e-01	[-9.5050214e-03]	 2.1631503e-01


.. parsed-literal::

       8	-4.4481990e-02	 2.4805948e-01	-2.4904880e-02	 2.4190025e-01	[ 5.9597278e-04]	 2.0716190e-01


.. parsed-literal::

       9	-3.3277029e-02	 2.4598285e-01	-1.6169157e-02	 2.3901098e-01	[ 1.3978011e-02]	 2.2198820e-01


.. parsed-literal::

      10	-2.5312858e-02	 2.4467630e-01	-1.0081821e-02	 2.3755180e-01	[ 2.0457055e-02]	 2.1506214e-01


.. parsed-literal::

      11	-1.8693244e-02	 2.4328819e-01	-4.2778806e-03	 2.3517488e-01	[ 3.2518907e-02]	 2.2460723e-01
      12	-1.6178724e-02	 2.4285458e-01	-1.9729558e-03	 2.3441429e-01	[ 3.4498968e-02]	 1.9868755e-01


.. parsed-literal::

      13	-1.1307062e-02	 2.4195714e-01	 2.7962356e-03	 2.3308591e-01	[ 4.0855002e-02]	 1.8005705e-01


.. parsed-literal::

      14	 4.6239296e-02	 2.3108811e-01	 6.2409395e-02	 2.2116868e-01	[ 9.3394162e-02]	 3.1926918e-01


.. parsed-literal::

      15	 7.1694051e-02	 2.2710832e-01	 8.9919798e-02	 2.1870118e-01	[ 1.0914813e-01]	 3.2394481e-01


.. parsed-literal::

      16	 1.2936249e-01	 2.2336128e-01	 1.4909498e-01	 2.1526839e-01	[ 1.6967878e-01]	 2.0868468e-01


.. parsed-literal::

      17	 2.5986403e-01	 2.1792555e-01	 2.9019672e-01	 2.1020688e-01	[ 3.2437894e-01]	 2.0654893e-01


.. parsed-literal::

      18	 2.9974609e-01	 2.1840360e-01	 3.3046937e-01	 2.1342670e-01	[ 3.5021731e-01]	 2.0191240e-01


.. parsed-literal::

      19	 3.3962440e-01	 2.1807011e-01	 3.7090590e-01	 2.0932087e-01	[ 4.0383806e-01]	 2.0315790e-01


.. parsed-literal::

      20	 3.8976815e-01	 2.1258526e-01	 4.2185524e-01	 2.0322133e-01	[ 4.5449238e-01]	 2.0479202e-01


.. parsed-literal::

      21	 4.7440159e-01	 2.0697147e-01	 5.0754284e-01	 1.9969659e-01	[ 5.3656225e-01]	 2.0844388e-01


.. parsed-literal::

      22	 5.5837437e-01	 2.0362953e-01	 5.9498314e-01	 1.9778086e-01	[ 6.1384426e-01]	 2.0640111e-01
      23	 6.1791259e-01	 1.9772140e-01	 6.5646668e-01	 1.9374968e-01	[ 6.6266397e-01]	 2.0489097e-01


.. parsed-literal::

      24	 6.5728912e-01	 1.9426051e-01	 6.9520501e-01	 1.8730960e-01	[ 7.0031179e-01]	 2.1670175e-01
      25	 7.1169031e-01	 1.9215146e-01	 7.4936141e-01	 1.8438373e-01	[ 7.5333687e-01]	 1.8615532e-01


.. parsed-literal::

      26	 7.5180217e-01	 1.9413044e-01	 7.8803420e-01	 1.8554098e-01	[ 7.8101932e-01]	 3.3146286e-01


.. parsed-literal::

      27	 7.7741132e-01	 1.9426466e-01	 8.1373946e-01	 1.8614518e-01	[ 8.0596399e-01]	 2.0807242e-01


.. parsed-literal::

      28	 8.0874313e-01	 1.9012018e-01	 8.4629985e-01	 1.8264658e-01	[ 8.3438517e-01]	 2.1185493e-01
      29	 8.5695303e-01	 1.8453802e-01	 8.9686823e-01	 1.7939335e-01	[ 8.9224011e-01]	 1.8721914e-01


.. parsed-literal::

      30	 8.7240803e-01	 1.8274757e-01	 9.1304541e-01	 1.7865009e-01	[ 9.1087749e-01]	 2.0322657e-01


.. parsed-literal::

      31	 8.9115750e-01	 1.8209073e-01	 9.3145772e-01	 1.7741677e-01	[ 9.2768618e-01]	 2.0874667e-01
      32	 9.0157966e-01	 1.8138444e-01	 9.4223163e-01	 1.7661589e-01	[ 9.3794268e-01]	 1.9478822e-01


.. parsed-literal::

      33	 9.2030360e-01	 1.8036351e-01	 9.6193965e-01	 1.7566609e-01	[ 9.5633213e-01]	 2.1025181e-01


.. parsed-literal::

      34	 9.4463232e-01	 1.7866037e-01	 9.8738895e-01	 1.7490151e-01	[ 9.7247650e-01]	 2.0599198e-01


.. parsed-literal::

      35	 9.5783316e-01	 1.7800350e-01	 1.0014985e+00	 1.7235783e-01	[ 9.7615998e-01]	 2.2240376e-01
      36	 9.7590413e-01	 1.7560614e-01	 1.0194757e+00	 1.7082542e-01	[ 9.9357774e-01]	 1.7480016e-01


.. parsed-literal::

      37	 9.8941846e-01	 1.7427479e-01	 1.0333291e+00	 1.6960158e-01	[ 1.0058274e+00]	 1.7374063e-01


.. parsed-literal::

      38	 1.0041447e+00	 1.7463756e-01	 1.0489801e+00	 1.6989823e-01	[ 1.0244103e+00]	 2.0452714e-01


.. parsed-literal::

      39	 1.0163765e+00	 1.7410026e-01	 1.0617970e+00	 1.6780017e-01	[ 1.0356098e+00]	 2.1321464e-01


.. parsed-literal::

      40	 1.0264246e+00	 1.7367160e-01	 1.0718761e+00	 1.6731617e-01	[ 1.0460985e+00]	 2.1900463e-01
      41	 1.0392714e+00	 1.7225921e-01	 1.0850507e+00	 1.6549921e-01	[ 1.0578714e+00]	 1.9904947e-01


.. parsed-literal::

      42	 1.0500555e+00	 1.7108339e-01	 1.0959679e+00	 1.6396952e-01	[ 1.0669426e+00]	 2.1707368e-01
      43	 1.0634056e+00	 1.6903319e-01	 1.1096352e+00	 1.6187306e-01	[ 1.0774955e+00]	 1.7619419e-01


.. parsed-literal::

      44	 1.0737271e+00	 1.6910792e-01	 1.1199615e+00	 1.6236700e-01	[ 1.0799974e+00]	 2.0721841e-01


.. parsed-literal::

      45	 1.0817338e+00	 1.6813002e-01	 1.1279350e+00	 1.6147950e-01	[ 1.0891004e+00]	 2.0686507e-01
      46	 1.0901215e+00	 1.6681286e-01	 1.1365499e+00	 1.6048530e-01	[ 1.0976120e+00]	 1.8906593e-01


.. parsed-literal::

      47	 1.1011500e+00	 1.6491442e-01	 1.1479432e+00	 1.5915650e-01	[ 1.1091490e+00]	 2.0519876e-01
      48	 1.1109756e+00	 1.6247396e-01	 1.1585717e+00	 1.5671798e-01	[ 1.1186352e+00]	 1.8773770e-01


.. parsed-literal::

      49	 1.1217814e+00	 1.6122824e-01	 1.1691906e+00	 1.5567009e-01	[ 1.1293327e+00]	 2.1139836e-01
      50	 1.1290778e+00	 1.6022599e-01	 1.1766377e+00	 1.5442746e-01	[ 1.1359496e+00]	 1.8753910e-01


.. parsed-literal::

      51	 1.1399782e+00	 1.5879067e-01	 1.1878180e+00	 1.5271699e-01	[ 1.1454170e+00]	 2.0924354e-01


.. parsed-literal::

      52	 1.1499252e+00	 1.5702531e-01	 1.1980468e+00	 1.5149232e-01	[ 1.1522431e+00]	 2.0651650e-01


.. parsed-literal::

      53	 1.1607197e+00	 1.5528399e-01	 1.2089757e+00	 1.5060516e-01	[ 1.1565299e+00]	 2.1187663e-01


.. parsed-literal::

      54	 1.1675758e+00	 1.5493156e-01	 1.2156691e+00	 1.5077436e-01	[ 1.1628042e+00]	 2.0787191e-01


.. parsed-literal::

      55	 1.1781377e+00	 1.5353648e-01	 1.2264783e+00	 1.5069192e-01	[ 1.1701416e+00]	 2.1493483e-01


.. parsed-literal::

      56	 1.1865118e+00	 1.5231800e-01	 1.2349951e+00	 1.4988758e-01	  1.1699122e+00 	 2.1370578e-01


.. parsed-literal::

      57	 1.1950554e+00	 1.5155511e-01	 1.2435350e+00	 1.4971589e-01	[ 1.1756984e+00]	 2.0678544e-01
      58	 1.2041355e+00	 1.5083356e-01	 1.2529063e+00	 1.4904912e-01	[ 1.1808648e+00]	 2.0010972e-01


.. parsed-literal::

      59	 1.2120630e+00	 1.5029128e-01	 1.2610829e+00	 1.4865988e-01	[ 1.1845151e+00]	 2.0955491e-01
      60	 1.2217881e+00	 1.4945098e-01	 1.2716162e+00	 1.4832100e-01	  1.1818884e+00 	 1.8628097e-01


.. parsed-literal::

      61	 1.2293360e+00	 1.4914365e-01	 1.2790399e+00	 1.4883758e-01	  1.1832273e+00 	 1.8232751e-01


.. parsed-literal::

      62	 1.2340223e+00	 1.4847926e-01	 1.2836631e+00	 1.4878354e-01	  1.1829715e+00 	 2.0951271e-01


.. parsed-literal::

      63	 1.2403426e+00	 1.4798062e-01	 1.2899249e+00	 1.4852355e-01	[ 1.1905579e+00]	 2.1679235e-01


.. parsed-literal::

      64	 1.2499392e+00	 1.4721553e-01	 1.3000022e+00	 1.4840074e-01	  1.1886260e+00 	 2.0833230e-01


.. parsed-literal::

      65	 1.2583963e+00	 1.4687267e-01	 1.3087040e+00	 1.4883465e-01	[ 1.2021010e+00]	 2.1214223e-01


.. parsed-literal::

      66	 1.2651987e+00	 1.4624883e-01	 1.3154049e+00	 1.4822725e-01	[ 1.2046992e+00]	 2.1336079e-01
      67	 1.2731742e+00	 1.4528964e-01	 1.3232775e+00	 1.4763056e-01	[ 1.2099753e+00]	 1.9562364e-01


.. parsed-literal::

      68	 1.2814095e+00	 1.4441567e-01	 1.3319667e+00	 1.4702382e-01	[ 1.2130778e+00]	 2.0967746e-01


.. parsed-literal::

      69	 1.2906162e+00	 1.4382280e-01	 1.3413448e+00	 1.4645377e-01	[ 1.2215221e+00]	 2.0178699e-01


.. parsed-literal::

      70	 1.2974984e+00	 1.4383880e-01	 1.3483091e+00	 1.4629177e-01	[ 1.2262269e+00]	 2.0416665e-01
      71	 1.3050646e+00	 1.4385259e-01	 1.3560528e+00	 1.4619710e-01	[ 1.2315404e+00]	 1.8067217e-01


.. parsed-literal::

      72	 1.3100334e+00	 1.4373041e-01	 1.3610445e+00	 1.4612055e-01	[ 1.2359725e+00]	 1.7524505e-01


.. parsed-literal::

      73	 1.3163256e+00	 1.4304545e-01	 1.3672533e+00	 1.4600044e-01	[ 1.2439487e+00]	 2.1519804e-01
      74	 1.3215298e+00	 1.4189927e-01	 1.3726520e+00	 1.4581261e-01	[ 1.2460467e+00]	 1.6569495e-01


.. parsed-literal::

      75	 1.3263331e+00	 1.4127837e-01	 1.3775408e+00	 1.4555887e-01	[ 1.2517525e+00]	 2.1468210e-01


.. parsed-literal::

      76	 1.3332012e+00	 1.4035996e-01	 1.3846054e+00	 1.4518211e-01	[ 1.2565382e+00]	 2.2013640e-01


.. parsed-literal::

      77	 1.3402720e+00	 1.3923781e-01	 1.3920771e+00	 1.4482570e-01	[ 1.2595863e+00]	 2.1180010e-01


.. parsed-literal::

      78	 1.3457752e+00	 1.3823624e-01	 1.3976944e+00	 1.4448885e-01	[ 1.2631055e+00]	 2.0401645e-01


.. parsed-literal::

      79	 1.3503880e+00	 1.3811793e-01	 1.4022277e+00	 1.4436160e-01	[ 1.2656794e+00]	 2.0465374e-01


.. parsed-literal::

      80	 1.3539022e+00	 1.3783329e-01	 1.4057533e+00	 1.4432940e-01	[ 1.2680799e+00]	 2.2020459e-01


.. parsed-literal::

      81	 1.3596626e+00	 1.3707496e-01	 1.4119228e+00	 1.4432889e-01	[ 1.2712557e+00]	 2.1963525e-01
      82	 1.3654423e+00	 1.3632316e-01	 1.4177471e+00	 1.4502645e-01	[ 1.2780444e+00]	 1.8505812e-01


.. parsed-literal::

      83	 1.3707696e+00	 1.3542422e-01	 1.4230866e+00	 1.4530673e-01	[ 1.2861772e+00]	 1.8197727e-01


.. parsed-literal::

      84	 1.3778073e+00	 1.3428751e-01	 1.4303319e+00	 1.4554165e-01	[ 1.2974545e+00]	 2.0311332e-01
      85	 1.3818131e+00	 1.3310142e-01	 1.4344697e+00	 1.4496695e-01	[ 1.3027077e+00]	 1.9596767e-01


.. parsed-literal::

      86	 1.3864472e+00	 1.3308560e-01	 1.4389649e+00	 1.4460096e-01	[ 1.3075977e+00]	 2.1271539e-01
      87	 1.3912075e+00	 1.3302561e-01	 1.4437573e+00	 1.4381827e-01	[ 1.3126659e+00]	 1.9935393e-01


.. parsed-literal::

      88	 1.3956145e+00	 1.3298221e-01	 1.4481635e+00	 1.4324969e-01	[ 1.3178877e+00]	 2.1072650e-01


.. parsed-literal::

      89	 1.4003077e+00	 1.3235736e-01	 1.4532894e+00	 1.4172605e-01	[ 1.3220551e+00]	 2.0853162e-01
      90	 1.4054874e+00	 1.3205772e-01	 1.4583407e+00	 1.4168142e-01	[ 1.3268202e+00]	 1.9980383e-01


.. parsed-literal::

      91	 1.4078298e+00	 1.3179411e-01	 1.4606084e+00	 1.4202925e-01	[ 1.3288590e+00]	 2.0864940e-01


.. parsed-literal::

      92	 1.4128974e+00	 1.3095690e-01	 1.4657607e+00	 1.4247495e-01	[ 1.3306343e+00]	 2.0869088e-01


.. parsed-literal::

      93	 1.4158008e+00	 1.3067295e-01	 1.4688543e+00	 1.4262838e-01	[ 1.3310370e+00]	 2.2104764e-01


.. parsed-literal::

      94	 1.4204907e+00	 1.3038016e-01	 1.4734917e+00	 1.4233102e-01	[ 1.3336329e+00]	 2.1264768e-01


.. parsed-literal::

      95	 1.4243904e+00	 1.3014138e-01	 1.4774335e+00	 1.4168593e-01	[ 1.3356844e+00]	 2.1087384e-01


.. parsed-literal::

      96	 1.4270938e+00	 1.3008156e-01	 1.4801403e+00	 1.4143561e-01	[ 1.3380636e+00]	 2.1128106e-01


.. parsed-literal::

      97	 1.4339522e+00	 1.2981120e-01	 1.4870942e+00	 1.4154103e-01	[ 1.3394027e+00]	 2.1054506e-01


.. parsed-literal::

      98	 1.4365722e+00	 1.2952976e-01	 1.4897256e+00	 1.4173083e-01	[ 1.3399314e+00]	 3.2388878e-01


.. parsed-literal::

      99	 1.4395149e+00	 1.2934703e-01	 1.4926399e+00	 1.4202368e-01	[ 1.3411369e+00]	 2.1056557e-01


.. parsed-literal::

     100	 1.4427728e+00	 1.2903776e-01	 1.4958632e+00	 1.4246688e-01	[ 1.3412524e+00]	 2.2037792e-01
     101	 1.4457918e+00	 1.2853696e-01	 1.4989021e+00	 1.4276823e-01	  1.3411863e+00 	 1.8670654e-01


.. parsed-literal::

     102	 1.4495665e+00	 1.2828202e-01	 1.5026477e+00	 1.4270150e-01	[ 1.3416022e+00]	 2.1444249e-01


.. parsed-literal::

     103	 1.4534563e+00	 1.2796759e-01	 1.5065504e+00	 1.4240480e-01	[ 1.3435498e+00]	 2.0913625e-01


.. parsed-literal::

     104	 1.4574810e+00	 1.2727662e-01	 1.5107469e+00	 1.4166213e-01	[ 1.3457278e+00]	 2.2114992e-01


.. parsed-literal::

     105	 1.4610069e+00	 1.2701143e-01	 1.5143555e+00	 1.4128722e-01	[ 1.3477963e+00]	 2.0903754e-01


.. parsed-literal::

     106	 1.4646271e+00	 1.2669082e-01	 1.5180657e+00	 1.4090667e-01	[ 1.3494893e+00]	 2.0081663e-01


.. parsed-literal::

     107	 1.4680253e+00	 1.2606756e-01	 1.5216809e+00	 1.4067043e-01	  1.3423927e+00 	 2.1111369e-01


.. parsed-literal::

     108	 1.4710437e+00	 1.2598004e-01	 1.5247452e+00	 1.4037867e-01	  1.3436817e+00 	 2.0498228e-01


.. parsed-literal::

     109	 1.4728295e+00	 1.2597542e-01	 1.5264438e+00	 1.4042464e-01	  1.3448940e+00 	 2.0852137e-01


.. parsed-literal::

     110	 1.4762775e+00	 1.2580920e-01	 1.5298606e+00	 1.4057333e-01	  1.3446264e+00 	 2.1538877e-01


.. parsed-literal::

     111	 1.4797671e+00	 1.2564809e-01	 1.5334168e+00	 1.4050243e-01	  1.3422561e+00 	 2.1060681e-01
     112	 1.4828546e+00	 1.2545607e-01	 1.5365761e+00	 1.4115780e-01	  1.3379820e+00 	 2.0001078e-01


.. parsed-literal::

     113	 1.4858564e+00	 1.2522248e-01	 1.5396437e+00	 1.4084560e-01	  1.3369208e+00 	 2.1008682e-01


.. parsed-literal::

     114	 1.4880751e+00	 1.2500627e-01	 1.5418926e+00	 1.4037015e-01	  1.3376489e+00 	 2.0980406e-01


.. parsed-literal::

     115	 1.4907077e+00	 1.2457893e-01	 1.5446182e+00	 1.3994618e-01	  1.3361694e+00 	 2.1258426e-01
     116	 1.4931258e+00	 1.2421732e-01	 1.5471943e+00	 1.3930955e-01	  1.3295365e+00 	 1.7584658e-01


.. parsed-literal::

     117	 1.4958398e+00	 1.2410357e-01	 1.5498259e+00	 1.3924631e-01	  1.3333520e+00 	 1.9933319e-01


.. parsed-literal::

     118	 1.4975441e+00	 1.2399645e-01	 1.5515211e+00	 1.3913938e-01	  1.3325798e+00 	 2.1787095e-01


.. parsed-literal::

     119	 1.4995319e+00	 1.2381007e-01	 1.5534822e+00	 1.3885562e-01	  1.3308571e+00 	 2.1106386e-01


.. parsed-literal::

     120	 1.5015925e+00	 1.2363381e-01	 1.5555174e+00	 1.3833297e-01	  1.3286709e+00 	 2.1252465e-01


.. parsed-literal::

     121	 1.5032413e+00	 1.2353142e-01	 1.5571429e+00	 1.3823295e-01	  1.3284160e+00 	 2.1259999e-01


.. parsed-literal::

     122	 1.5047046e+00	 1.2341359e-01	 1.5586042e+00	 1.3810530e-01	  1.3287072e+00 	 2.0608187e-01


.. parsed-literal::

     123	 1.5068138e+00	 1.2321929e-01	 1.5607588e+00	 1.3781074e-01	  1.3273998e+00 	 2.1290636e-01


.. parsed-literal::

     124	 1.5074699e+00	 1.2284430e-01	 1.5616300e+00	 1.3782906e-01	  1.3230456e+00 	 2.0718527e-01


.. parsed-literal::

     125	 1.5106992e+00	 1.2282951e-01	 1.5647972e+00	 1.3751944e-01	  1.3229483e+00 	 2.1129680e-01


.. parsed-literal::

     126	 1.5118295e+00	 1.2279741e-01	 1.5659387e+00	 1.3743790e-01	  1.3226921e+00 	 2.1542239e-01


.. parsed-literal::

     127	 1.5136884e+00	 1.2267585e-01	 1.5678579e+00	 1.3731658e-01	  1.3215636e+00 	 2.0978141e-01
     128	 1.5161965e+00	 1.2255894e-01	 1.5704023e+00	 1.3718778e-01	  1.3228683e+00 	 1.8113399e-01


.. parsed-literal::

     129	 1.5175575e+00	 1.2244861e-01	 1.5718112e+00	 1.3708321e-01	  1.3236022e+00 	 3.3113718e-01


.. parsed-literal::

     130	 1.5197626e+00	 1.2241477e-01	 1.5740078e+00	 1.3694819e-01	  1.3264740e+00 	 2.0959234e-01
     131	 1.5210565e+00	 1.2237021e-01	 1.5752776e+00	 1.3673261e-01	  1.3282850e+00 	 1.8596411e-01


.. parsed-literal::

     132	 1.5225692e+00	 1.2232247e-01	 1.5768109e+00	 1.3629915e-01	  1.3295959e+00 	 1.9849062e-01


.. parsed-literal::

     133	 1.5240086e+00	 1.2225692e-01	 1.5783016e+00	 1.3564370e-01	  1.3280172e+00 	 2.2051287e-01


.. parsed-literal::

     134	 1.5255223e+00	 1.2223012e-01	 1.5798779e+00	 1.3510408e-01	  1.3262782e+00 	 2.0190501e-01


.. parsed-literal::

     135	 1.5271567e+00	 1.2222726e-01	 1.5816266e+00	 1.3457117e-01	  1.3227818e+00 	 2.1060014e-01
     136	 1.5287250e+00	 1.2218310e-01	 1.5832860e+00	 1.3425863e-01	  1.3206977e+00 	 1.8994427e-01


.. parsed-literal::

     137	 1.5308369e+00	 1.2217126e-01	 1.5854979e+00	 1.3409782e-01	  1.3186811e+00 	 2.0305371e-01
     138	 1.5330734e+00	 1.2206142e-01	 1.5877364e+00	 1.3415737e-01	  1.3206716e+00 	 1.9785452e-01


.. parsed-literal::

     139	 1.5353183e+00	 1.2198968e-01	 1.5899417e+00	 1.3441782e-01	  1.3210365e+00 	 2.1201587e-01


.. parsed-literal::

     140	 1.5370670e+00	 1.2182375e-01	 1.5916690e+00	 1.3435791e-01	  1.3235968e+00 	 2.1829414e-01
     141	 1.5385348e+00	 1.2171324e-01	 1.5931062e+00	 1.3456001e-01	  1.3224668e+00 	 1.9815302e-01


.. parsed-literal::

     142	 1.5397291e+00	 1.2161791e-01	 1.5943640e+00	 1.3449224e-01	  1.3191450e+00 	 2.0836306e-01
     143	 1.5409612e+00	 1.2155051e-01	 1.5956955e+00	 1.3439997e-01	  1.3156293e+00 	 2.0078158e-01


.. parsed-literal::

     144	 1.5420737e+00	 1.2145351e-01	 1.5969375e+00	 1.3430739e-01	  1.3098853e+00 	 2.0821309e-01
     145	 1.5436771e+00	 1.2141996e-01	 1.5985311e+00	 1.3456045e-01	  1.3107412e+00 	 1.8413377e-01


.. parsed-literal::

     146	 1.5449504e+00	 1.2138758e-01	 1.5997597e+00	 1.3490612e-01	  1.3118264e+00 	 2.1107507e-01


.. parsed-literal::

     147	 1.5460174e+00	 1.2134666e-01	 1.6007961e+00	 1.3519945e-01	  1.3115698e+00 	 2.1790981e-01
     148	 1.5474144e+00	 1.2121171e-01	 1.6021581e+00	 1.3553399e-01	  1.3072225e+00 	 1.9658923e-01


.. parsed-literal::

     149	 1.5488458e+00	 1.2116616e-01	 1.6035538e+00	 1.3550739e-01	  1.3076510e+00 	 2.0086718e-01
     150	 1.5500471e+00	 1.2108492e-01	 1.6047630e+00	 1.3533278e-01	  1.3056784e+00 	 1.9643593e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.23 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2d1ca68be0>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.12 s, sys: 32 ms, total: 2.15 s
    Wall time: 652 ms


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

