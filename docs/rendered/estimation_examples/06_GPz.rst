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
       1	-3.4795431e-01	 3.2213720e-01	-3.3821783e-01	 3.1530001e-01	[-3.2420619e-01]	 4.6450043e-01


.. parsed-literal::

       2	-2.7871769e-01	 3.1205302e-01	-2.5552254e-01	 3.0358671e-01	[-2.3055390e-01]	 2.3080730e-01


.. parsed-literal::

       3	-2.3479191e-01	 2.9161544e-01	-1.9394079e-01	 2.8446954e-01	[-1.6157750e-01]	 2.9129028e-01
       4	-2.0020269e-01	 2.6805755e-01	-1.5849139e-01	 2.6092516e-01	[-1.1333160e-01]	 1.7161155e-01


.. parsed-literal::

       5	-1.1473543e-01	 2.5961218e-01	-8.1013956e-02	 2.5245201e-01	[-4.6570739e-02]	 1.8942404e-01


.. parsed-literal::

       6	-7.8120477e-02	 2.5342657e-01	-4.7626276e-02	 2.4625665e-01	[-2.1700216e-02]	 2.0864606e-01
       7	-6.1086004e-02	 2.5095429e-01	-3.6776709e-02	 2.4344521e-01	[-7.8503872e-03]	 1.8585777e-01


.. parsed-literal::

       8	-4.5762557e-02	 2.4826033e-01	-2.5641934e-02	 2.4004718e-01	[ 6.5986134e-03]	 2.0973444e-01


.. parsed-literal::

       9	-3.1601852e-02	 2.4559947e-01	-1.4430172e-02	 2.3757938e-01	[ 1.8231718e-02]	 2.1629119e-01
      10	-2.3052688e-02	 2.4416706e-01	-8.1082839e-03	 2.3592266e-01	[ 2.3523598e-02]	 1.9952393e-01


.. parsed-literal::

      11	-1.6122687e-02	 2.4280320e-01	-1.8435428e-03	 2.3480127e-01	[ 3.0780800e-02]	 1.8453193e-01
      12	-1.3507048e-02	 2.4233820e-01	 5.1340346e-04	 2.3457519e-01	[ 3.1468438e-02]	 1.9382429e-01


.. parsed-literal::

      13	-9.3578997e-03	 2.4154971e-01	 4.6389321e-03	 2.3381680e-01	[ 3.5459534e-02]	 1.9936585e-01


.. parsed-literal::

      14	 4.7100689e-02	 2.2915930e-01	 6.2958148e-02	 2.2484336e-01	[ 7.8146280e-02]	 2.9028392e-01


.. parsed-literal::

      15	 7.4404985e-02	 2.2514370e-01	 9.4010710e-02	 2.2550637e-01	[ 1.0582757e-01]	 2.1202946e-01
      16	 1.6373629e-01	 2.2099874e-01	 1.8512879e-01	 2.1873796e-01	[ 1.9889626e-01]	 1.9923091e-01


.. parsed-literal::

      17	 2.6258745e-01	 2.1661089e-01	 2.9322215e-01	 2.1173106e-01	[ 3.2369277e-01]	 2.1473455e-01


.. parsed-literal::

      18	 2.9894111e-01	 2.1853603e-01	 3.3043894e-01	 2.1509491e-01	[ 3.6194933e-01]	 2.0802903e-01


.. parsed-literal::

      19	 3.3684331e-01	 2.1740745e-01	 3.6737689e-01	 2.1487011e-01	[ 3.9546332e-01]	 2.2581315e-01
      20	 3.8342462e-01	 2.1548274e-01	 4.1448365e-01	 2.1040589e-01	[ 4.4230978e-01]	 1.9435716e-01


.. parsed-literal::

      21	 4.5080669e-01	 2.1244468e-01	 4.8335755e-01	 2.0567589e-01	[ 5.1654822e-01]	 2.1163249e-01
      22	 5.2561902e-01	 2.1386643e-01	 5.6131962e-01	 2.0738793e-01	[ 6.0130972e-01]	 1.7231202e-01


.. parsed-literal::

      23	 5.7831892e-01	 2.0854130e-01	 6.1729419e-01	 2.0240288e-01	[ 6.5971962e-01]	 2.0780611e-01


.. parsed-literal::

      24	 6.1085845e-01	 2.0403331e-01	 6.4920045e-01	 1.9816192e-01	[ 6.8401543e-01]	 2.1814871e-01


.. parsed-literal::

      25	 6.4836648e-01	 1.9977539e-01	 6.8609426e-01	 1.9403690e-01	[ 7.2143903e-01]	 2.2788143e-01


.. parsed-literal::

      26	 7.0464424e-01	 1.9830423e-01	 7.4126657e-01	 1.9019016e-01	[ 7.8108036e-01]	 2.1565199e-01


.. parsed-literal::

      27	 7.4310494e-01	 2.0009552e-01	 7.7981659e-01	 1.9092799e-01	[ 8.0444504e-01]	 2.2011590e-01


.. parsed-literal::

      28	 7.7765013e-01	 1.9692200e-01	 8.1784380e-01	 1.8904125e-01	[ 8.2442780e-01]	 2.0611739e-01


.. parsed-literal::

      29	 8.0944612e-01	 1.9392450e-01	 8.4944120e-01	 1.8712393e-01	[ 8.5675411e-01]	 2.1858811e-01


.. parsed-literal::

      30	 8.3853930e-01	 1.9209728e-01	 8.7901204e-01	 1.8546127e-01	[ 8.7817063e-01]	 2.0125365e-01


.. parsed-literal::

      31	 8.6439022e-01	 1.9016508e-01	 9.0471466e-01	 1.8461568e-01	[ 9.0624651e-01]	 2.1477866e-01


.. parsed-literal::

      32	 8.9356214e-01	 1.8683117e-01	 9.3473553e-01	 1.8270343e-01	[ 9.3168600e-01]	 2.1004581e-01


.. parsed-literal::

      33	 9.2799807e-01	 1.8352347e-01	 9.7099871e-01	 1.7529612e-01	[ 9.4918491e-01]	 2.1925449e-01


.. parsed-literal::

      34	 9.5868694e-01	 1.7802643e-01	 1.0024768e+00	 1.7245725e-01	[ 9.6918034e-01]	 2.1267462e-01


.. parsed-literal::

      35	 9.7442355e-01	 1.7693821e-01	 1.0178680e+00	 1.7117686e-01	[ 9.8848423e-01]	 2.1487713e-01
      36	 9.8844895e-01	 1.7611548e-01	 1.0323892e+00	 1.7011556e-01	[ 1.0007632e+00]	 1.9747591e-01


.. parsed-literal::

      37	 1.0044344e+00	 1.7482930e-01	 1.0492690e+00	 1.6825259e-01	[ 1.0159356e+00]	 2.0130324e-01


.. parsed-literal::

      38	 1.0122398e+00	 1.7417538e-01	 1.0578153e+00	 1.6756853e-01	[ 1.0247243e+00]	 2.0368886e-01


.. parsed-literal::

      39	 1.0227649e+00	 1.7321062e-01	 1.0680854e+00	 1.6673623e-01	[ 1.0358683e+00]	 2.0971894e-01
      40	 1.0339538e+00	 1.7192331e-01	 1.0797319e+00	 1.6581954e-01	[ 1.0450024e+00]	 1.8585587e-01


.. parsed-literal::

      41	 1.0409357e+00	 1.7136277e-01	 1.0869504e+00	 1.6545855e-01	[ 1.0507897e+00]	 2.0796871e-01


.. parsed-literal::

      42	 1.0527313e+00	 1.7087207e-01	 1.0993145e+00	 1.6471166e-01	[ 1.0635740e+00]	 2.1038175e-01
      43	 1.0652456e+00	 1.7049441e-01	 1.1123821e+00	 1.6425932e-01	[ 1.0730959e+00]	 1.9335437e-01


.. parsed-literal::

      44	 1.0786017e+00	 1.7072137e-01	 1.1256913e+00	 1.6374615e-01	[ 1.0847128e+00]	 2.1231651e-01


.. parsed-literal::

      45	 1.0913292e+00	 1.6986905e-01	 1.1386082e+00	 1.6225733e-01	[ 1.0902720e+00]	 2.0196486e-01


.. parsed-literal::

      46	 1.1048628e+00	 1.6850996e-01	 1.1518658e+00	 1.6192075e-01	[ 1.0998163e+00]	 2.0688891e-01
      47	 1.1146145e+00	 1.6683385e-01	 1.1613838e+00	 1.6058219e-01	[ 1.1115970e+00]	 1.6406274e-01


.. parsed-literal::

      48	 1.1268434e+00	 1.6536258e-01	 1.1741479e+00	 1.5868120e-01	[ 1.1233935e+00]	 1.8288112e-01


.. parsed-literal::

      49	 1.1395567e+00	 1.6427246e-01	 1.1872314e+00	 1.5683817e-01	[ 1.1346121e+00]	 2.1179128e-01
      50	 1.1483563e+00	 1.6357567e-01	 1.1962744e+00	 1.5596492e-01	[ 1.1422076e+00]	 1.8429375e-01


.. parsed-literal::

      51	 1.1552079e+00	 1.6344676e-01	 1.2032769e+00	 1.5529706e-01	[ 1.1469708e+00]	 2.0314407e-01


.. parsed-literal::

      52	 1.1694219e+00	 1.6192646e-01	 1.2183195e+00	 1.5292112e-01	[ 1.1479653e+00]	 2.1670794e-01


.. parsed-literal::

      53	 1.1821274e+00	 1.6014869e-01	 1.2314635e+00	 1.5087346e-01	[ 1.1497867e+00]	 2.1114731e-01


.. parsed-literal::

      54	 1.1903219e+00	 1.5856286e-01	 1.2402302e+00	 1.4935545e-01	[ 1.1510984e+00]	 2.1124816e-01
      55	 1.1997439e+00	 1.5807575e-01	 1.2492225e+00	 1.4866459e-01	[ 1.1617534e+00]	 2.0096684e-01


.. parsed-literal::

      56	 1.2066991e+00	 1.5753206e-01	 1.2559533e+00	 1.4774535e-01	[ 1.1710754e+00]	 2.0152903e-01
      57	 1.2178585e+00	 1.5670673e-01	 1.2674422e+00	 1.4611725e-01	[ 1.1723627e+00]	 1.9911098e-01


.. parsed-literal::

      58	 1.2280475e+00	 1.5525894e-01	 1.2775424e+00	 1.4396612e-01	[ 1.1908135e+00]	 2.0732069e-01
      59	 1.2364016e+00	 1.5501753e-01	 1.2861208e+00	 1.4362035e-01	[ 1.1958886e+00]	 1.9636822e-01


.. parsed-literal::

      60	 1.2459796e+00	 1.5412030e-01	 1.2963978e+00	 1.4279943e-01	  1.1944202e+00 	 2.0192885e-01
      61	 1.2533884e+00	 1.5374760e-01	 1.3038039e+00	 1.4267463e-01	[ 1.1979411e+00]	 1.8434143e-01


.. parsed-literal::

      62	 1.2584393e+00	 1.5342639e-01	 1.3087270e+00	 1.4267828e-01	[ 1.2022191e+00]	 2.1264744e-01


.. parsed-literal::

      63	 1.2668490e+00	 1.5322317e-01	 1.3174971e+00	 1.4254188e-01	[ 1.2077331e+00]	 2.1033740e-01


.. parsed-literal::

      64	 1.2723242e+00	 1.5399792e-01	 1.3231256e+00	 1.4308661e-01	  1.2068332e+00 	 2.1975899e-01


.. parsed-literal::

      65	 1.2766344e+00	 1.5351295e-01	 1.3273075e+00	 1.4276910e-01	[ 1.2144310e+00]	 2.1500039e-01


.. parsed-literal::

      66	 1.2879570e+00	 1.5270406e-01	 1.3387876e+00	 1.4231234e-01	[ 1.2276002e+00]	 2.1836138e-01


.. parsed-literal::

      67	 1.2957057e+00	 1.5190863e-01	 1.3466839e+00	 1.4160267e-01	[ 1.2327552e+00]	 2.1002507e-01


.. parsed-literal::

      68	 1.3035809e+00	 1.5113415e-01	 1.3549212e+00	 1.4020594e-01	[ 1.2371432e+00]	 2.0683312e-01


.. parsed-literal::

      69	 1.3118301e+00	 1.4980200e-01	 1.3631824e+00	 1.3950161e-01	[ 1.2386950e+00]	 2.1201324e-01


.. parsed-literal::

      70	 1.3164578e+00	 1.4931390e-01	 1.3676462e+00	 1.3854107e-01	[ 1.2439245e+00]	 2.1451116e-01
      71	 1.3249971e+00	 1.4868970e-01	 1.3763616e+00	 1.3686896e-01	[ 1.2464277e+00]	 2.0289397e-01


.. parsed-literal::

      72	 1.3324890e+00	 1.4814052e-01	 1.3841890e+00	 1.3472509e-01	[ 1.2474283e+00]	 2.0320010e-01


.. parsed-literal::

      73	 1.3395958e+00	 1.4828480e-01	 1.3912481e+00	 1.3450683e-01	[ 1.2515613e+00]	 2.1183586e-01


.. parsed-literal::

      74	 1.3442525e+00	 1.4822462e-01	 1.3958640e+00	 1.3458341e-01	[ 1.2557124e+00]	 2.0645571e-01
      75	 1.3510988e+00	 1.4760761e-01	 1.4030687e+00	 1.3425799e-01	  1.2552551e+00 	 1.7208147e-01


.. parsed-literal::

      76	 1.3545807e+00	 1.4795597e-01	 1.4067456e+00	 1.3459446e-01	[ 1.2614235e+00]	 2.0890474e-01


.. parsed-literal::

      77	 1.3586333e+00	 1.4719202e-01	 1.4107052e+00	 1.3399135e-01	[ 1.2667965e+00]	 2.0625138e-01


.. parsed-literal::

      78	 1.3647772e+00	 1.4596351e-01	 1.4168714e+00	 1.3304963e-01	[ 1.2739673e+00]	 2.0952821e-01
      79	 1.3703212e+00	 1.4518407e-01	 1.4225187e+00	 1.3249799e-01	[ 1.2814246e+00]	 1.9982910e-01


.. parsed-literal::

      80	 1.3751531e+00	 1.4463201e-01	 1.4274499e+00	 1.3194214e-01	[ 1.2869719e+00]	 3.1839061e-01


.. parsed-literal::

      81	 1.3807223e+00	 1.4444383e-01	 1.4332089e+00	 1.3178626e-01	[ 1.2935204e+00]	 2.0523000e-01
      82	 1.3845411e+00	 1.4447659e-01	 1.4369641e+00	 1.3186490e-01	[ 1.2943522e+00]	 1.7657280e-01


.. parsed-literal::

      83	 1.3889057e+00	 1.4448515e-01	 1.4414028e+00	 1.3153609e-01	[ 1.2974702e+00]	 2.1527719e-01


.. parsed-literal::

      84	 1.3934637e+00	 1.4410590e-01	 1.4460237e+00	 1.3088607e-01	[ 1.2989375e+00]	 2.0114851e-01


.. parsed-literal::

      85	 1.3980207e+00	 1.4327705e-01	 1.4506490e+00	 1.3002990e-01	[ 1.3002022e+00]	 2.0703030e-01
      86	 1.4021464e+00	 1.4242913e-01	 1.4547591e+00	 1.2934312e-01	[ 1.3052685e+00]	 1.7364359e-01


.. parsed-literal::

      87	 1.4063198e+00	 1.4159432e-01	 1.4589569e+00	 1.2883678e-01	[ 1.3098107e+00]	 1.9615078e-01
      88	 1.4108317e+00	 1.4098091e-01	 1.4635417e+00	 1.2862630e-01	[ 1.3149780e+00]	 2.0951176e-01


.. parsed-literal::

      89	 1.4153719e+00	 1.4058157e-01	 1.4681977e+00	 1.2841881e-01	[ 1.3177410e+00]	 2.1450377e-01


.. parsed-literal::

      90	 1.4185945e+00	 1.4091382e-01	 1.4715147e+00	 1.2853756e-01	[ 1.3177544e+00]	 2.0542192e-01


.. parsed-literal::

      91	 1.4208319e+00	 1.4101093e-01	 1.4739067e+00	 1.2873481e-01	  1.3128546e+00 	 2.1211815e-01
      92	 1.4240375e+00	 1.4099538e-01	 1.4771508e+00	 1.2872285e-01	  1.3138573e+00 	 1.8445945e-01


.. parsed-literal::

      93	 1.4270718e+00	 1.4081952e-01	 1.4802043e+00	 1.2868966e-01	  1.3133385e+00 	 2.0321798e-01


.. parsed-literal::

      94	 1.4299288e+00	 1.4045953e-01	 1.4830815e+00	 1.2865778e-01	  1.3117795e+00 	 2.0776057e-01
      95	 1.4357648e+00	 1.3945132e-01	 1.4889911e+00	 1.2870510e-01	  1.3102817e+00 	 2.0413232e-01


.. parsed-literal::

      96	 1.4391058e+00	 1.3922368e-01	 1.4924045e+00	 1.2879674e-01	  1.3082261e+00 	 3.1600547e-01
      97	 1.4418459e+00	 1.3881297e-01	 1.4951749e+00	 1.2872841e-01	  1.3109098e+00 	 1.8565774e-01


.. parsed-literal::

      98	 1.4444843e+00	 1.3850601e-01	 1.4978667e+00	 1.2845286e-01	  1.3142515e+00 	 2.1478081e-01


.. parsed-literal::

      99	 1.4477403e+00	 1.3835973e-01	 1.5011919e+00	 1.2824267e-01	  1.3138330e+00 	 2.0387864e-01


.. parsed-literal::

     100	 1.4504222e+00	 1.3778308e-01	 1.5039534e+00	 1.2772803e-01	  1.3097381e+00 	 2.0398355e-01


.. parsed-literal::

     101	 1.4533820e+00	 1.3798617e-01	 1.5067945e+00	 1.2761043e-01	  1.3122694e+00 	 2.0114470e-01
     102	 1.4552468e+00	 1.3797255e-01	 1.5085987e+00	 1.2769083e-01	  1.3105470e+00 	 1.9990325e-01


.. parsed-literal::

     103	 1.4582013e+00	 1.3780889e-01	 1.5115138e+00	 1.2757018e-01	  1.3098642e+00 	 2.1738005e-01


.. parsed-literal::

     104	 1.4631217e+00	 1.3758413e-01	 1.5165471e+00	 1.2733306e-01	  1.3064161e+00 	 2.0886636e-01


.. parsed-literal::

     105	 1.4654659e+00	 1.3761539e-01	 1.5189166e+00	 1.2712642e-01	  1.3033315e+00 	 3.2389402e-01
     106	 1.4679677e+00	 1.3757507e-01	 1.5214528e+00	 1.2683368e-01	  1.3051519e+00 	 1.9473910e-01


.. parsed-literal::

     107	 1.4701493e+00	 1.3756532e-01	 1.5236870e+00	 1.2661829e-01	  1.3052001e+00 	 2.2020245e-01
     108	 1.4722989e+00	 1.3752534e-01	 1.5258916e+00	 1.2631605e-01	  1.2980700e+00 	 1.7516685e-01


.. parsed-literal::

     109	 1.4748357e+00	 1.3741616e-01	 1.5283613e+00	 1.2626005e-01	  1.2968847e+00 	 2.0876861e-01
     110	 1.4772797e+00	 1.3726821e-01	 1.5307410e+00	 1.2630525e-01	  1.2909108e+00 	 1.7540598e-01


.. parsed-literal::

     111	 1.4791828e+00	 1.3707397e-01	 1.5326279e+00	 1.2630420e-01	  1.2899883e+00 	 2.0387650e-01


.. parsed-literal::

     112	 1.4820078e+00	 1.3689145e-01	 1.5355463e+00	 1.2626179e-01	  1.2852598e+00 	 2.1097231e-01
     113	 1.4842287e+00	 1.3671272e-01	 1.5378213e+00	 1.2615624e-01	  1.2884327e+00 	 2.0013356e-01


.. parsed-literal::

     114	 1.4856268e+00	 1.3671650e-01	 1.5392263e+00	 1.2597578e-01	  1.2928289e+00 	 2.1600485e-01


.. parsed-literal::

     115	 1.4868215e+00	 1.3636078e-01	 1.5405384e+00	 1.2581079e-01	  1.2899202e+00 	 2.1100163e-01


.. parsed-literal::

     116	 1.4886541e+00	 1.3650827e-01	 1.5423281e+00	 1.2577041e-01	  1.2914088e+00 	 2.0713949e-01
     117	 1.4901023e+00	 1.3653143e-01	 1.5437807e+00	 1.2577108e-01	  1.2888860e+00 	 2.0174384e-01


.. parsed-literal::

     118	 1.4915628e+00	 1.3643608e-01	 1.5452643e+00	 1.2576341e-01	  1.2853146e+00 	 1.8668032e-01
     119	 1.4940702e+00	 1.3626063e-01	 1.5478069e+00	 1.2565306e-01	  1.2834035e+00 	 1.9786215e-01


.. parsed-literal::

     120	 1.4953965e+00	 1.3612676e-01	 1.5492255e+00	 1.2555519e-01	  1.2778924e+00 	 3.2973814e-01


.. parsed-literal::

     121	 1.4975760e+00	 1.3586031e-01	 1.5514541e+00	 1.2536860e-01	  1.2780955e+00 	 2.1283698e-01


.. parsed-literal::

     122	 1.4988339e+00	 1.3571758e-01	 1.5527367e+00	 1.2520605e-01	  1.2781408e+00 	 2.1366239e-01


.. parsed-literal::

     123	 1.5009749e+00	 1.3548195e-01	 1.5549554e+00	 1.2491946e-01	  1.2724136e+00 	 2.1763515e-01
     124	 1.5029231e+00	 1.3526751e-01	 1.5569550e+00	 1.2470784e-01	  1.2669913e+00 	 1.9872618e-01


.. parsed-literal::

     125	 1.5051628e+00	 1.3510987e-01	 1.5592147e+00	 1.2461944e-01	  1.2574690e+00 	 1.9738674e-01
     126	 1.5072434e+00	 1.3504948e-01	 1.5612472e+00	 1.2468726e-01	  1.2541535e+00 	 1.7831540e-01


.. parsed-literal::

     127	 1.5095931e+00	 1.3486586e-01	 1.5635568e+00	 1.2477229e-01	  1.2524580e+00 	 2.0881605e-01


.. parsed-literal::

     128	 1.5108703e+00	 1.3464920e-01	 1.5648786e+00	 1.2481771e-01	  1.2478845e+00 	 2.1061134e-01
     129	 1.5126329e+00	 1.3460058e-01	 1.5666162e+00	 1.2470524e-01	  1.2510672e+00 	 2.0353532e-01


.. parsed-literal::

     130	 1.5140070e+00	 1.3441765e-01	 1.5680454e+00	 1.2449956e-01	  1.2509805e+00 	 2.0810270e-01


.. parsed-literal::

     131	 1.5152796e+00	 1.3428524e-01	 1.5693526e+00	 1.2431326e-01	  1.2501432e+00 	 2.0256186e-01


.. parsed-literal::

     132	 1.5160402e+00	 1.3377317e-01	 1.5702754e+00	 1.2391568e-01	  1.2355834e+00 	 2.0757079e-01


.. parsed-literal::

     133	 1.5184515e+00	 1.3376243e-01	 1.5725870e+00	 1.2397333e-01	  1.2408080e+00 	 2.1635675e-01


.. parsed-literal::

     134	 1.5193004e+00	 1.3375391e-01	 1.5733841e+00	 1.2400628e-01	  1.2404556e+00 	 2.0461917e-01


.. parsed-literal::

     135	 1.5205220e+00	 1.3348790e-01	 1.5745858e+00	 1.2404706e-01	  1.2346406e+00 	 2.0785928e-01


.. parsed-literal::

     136	 1.5213641e+00	 1.3318990e-01	 1.5754877e+00	 1.2390884e-01	  1.2183075e+00 	 2.1370482e-01


.. parsed-literal::

     137	 1.5229023e+00	 1.3300302e-01	 1.5769852e+00	 1.2392991e-01	  1.2199057e+00 	 2.0393395e-01
     138	 1.5238521e+00	 1.3278238e-01	 1.5779567e+00	 1.2390187e-01	  1.2179623e+00 	 1.7268276e-01


.. parsed-literal::

     139	 1.5246329e+00	 1.3261560e-01	 1.5787671e+00	 1.2384888e-01	  1.2151733e+00 	 2.1378589e-01


.. parsed-literal::

     140	 1.5264453e+00	 1.3239811e-01	 1.5806328e+00	 1.2373419e-01	  1.2094887e+00 	 2.1388197e-01


.. parsed-literal::

     141	 1.5289198e+00	 1.3217588e-01	 1.5832192e+00	 1.2377182e-01	  1.2012018e+00 	 2.1462178e-01


.. parsed-literal::

     142	 1.5302939e+00	 1.3197745e-01	 1.5847667e+00	 1.2380138e-01	  1.1789562e+00 	 2.0823359e-01
     143	 1.5322041e+00	 1.3209342e-01	 1.5865785e+00	 1.2378768e-01	  1.1877341e+00 	 1.9830656e-01


.. parsed-literal::

     144	 1.5332764e+00	 1.3210276e-01	 1.5876115e+00	 1.2384690e-01	  1.1872198e+00 	 1.8279934e-01


.. parsed-literal::

     145	 1.5344971e+00	 1.3200415e-01	 1.5888848e+00	 1.2393669e-01	  1.1875381e+00 	 2.0270181e-01
     146	 1.5361377e+00	 1.3177341e-01	 1.5906026e+00	 1.2400227e-01	  1.1751408e+00 	 1.8256330e-01


.. parsed-literal::

     147	 1.5376072e+00	 1.3154479e-01	 1.5921248e+00	 1.2404093e-01	  1.1711424e+00 	 2.0372677e-01
     148	 1.5388952e+00	 1.3123420e-01	 1.5935058e+00	 1.2394312e-01	  1.1679246e+00 	 1.8332505e-01


.. parsed-literal::

     149	 1.5400188e+00	 1.3115312e-01	 1.5946421e+00	 1.2396025e-01	  1.1697330e+00 	 2.1549249e-01


.. parsed-literal::

     150	 1.5411562e+00	 1.3113854e-01	 1.5957873e+00	 1.2398832e-01	  1.1704120e+00 	 2.1503925e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.13 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f6e38b6ef20>



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
    CPU times: user 1.75 s, sys: 43.9 ms, total: 1.79 s
    Wall time: 598 ms


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

