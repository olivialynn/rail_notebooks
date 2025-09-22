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
       1	-3.3959605e-01	 3.1967728e-01	-3.2987265e-01	 3.2381249e-01	[-3.3816431e-01]	 4.6458125e-01


.. parsed-literal::

       2	-2.7063730e-01	 3.0941875e-01	-2.4723721e-01	 3.1422677e-01	[-2.6223256e-01]	 2.3114204e-01


.. parsed-literal::

       3	-2.2932501e-01	 2.9046950e-01	-1.8941710e-01	 2.9383819e-01	[-2.0571086e-01]	 3.0357504e-01
       4	-1.8535819e-01	 2.6616706e-01	-1.4309024e-01	 2.6511387e-01	[-1.3954732e-01]	 1.8401837e-01


.. parsed-literal::

       5	-1.0175627e-01	 2.5556638e-01	-6.8805258e-02	 2.5615007e-01	[-7.2450608e-02]	 2.1411395e-01


.. parsed-literal::

       6	-6.7875178e-02	 2.5114144e-01	-3.8088255e-02	 2.5290765e-01	[-4.3945515e-02]	 2.1602583e-01


.. parsed-literal::

       7	-4.9443267e-02	 2.4802775e-01	-2.5278206e-02	 2.4996239e-01	[-3.2168874e-02]	 2.1832013e-01
       8	-3.7386421e-02	 2.4611135e-01	-1.7186035e-02	 2.4822722e-01	[-2.5221214e-02]	 1.9995546e-01


.. parsed-literal::

       9	-2.5473273e-02	 2.4392069e-01	-7.9031127e-03	 2.4596572e-01	[-1.5652880e-02]	 2.1227217e-01


.. parsed-literal::

      10	-1.8553692e-02	 2.4181752e-01	-3.5969619e-03	 2.4372165e-01	[-1.2084471e-02]	 2.1452928e-01


.. parsed-literal::

      11	-9.0362884e-03	 2.4095285e-01	 5.4792421e-03	 2.4304189e-01	[-2.6521045e-03]	 2.1298242e-01


.. parsed-literal::

      12	-6.4919843e-03	 2.4047741e-01	 7.7220606e-03	 2.4293183e-01	[-1.9001780e-03]	 2.1714425e-01


.. parsed-literal::

      13	-3.1600523e-03	 2.3988438e-01	 1.0806201e-02	 2.4264097e-01	[ 4.6452470e-07]	 2.1631336e-01


.. parsed-literal::

      14	 8.7315154e-02	 2.2636667e-01	 1.0878405e-01	 2.2802691e-01	[ 1.0961654e-01]	 4.1037893e-01
      15	 1.6225270e-01	 2.2095773e-01	 1.8488041e-01	 2.2340225e-01	[ 1.8551133e-01]	 1.9896197e-01


.. parsed-literal::

      16	 2.1801407e-01	 2.1661511e-01	 2.4739535e-01	 2.1961052e-01	[ 2.4833504e-01]	 2.1263194e-01
      17	 2.8203473e-01	 2.1390821e-01	 3.1338552e-01	 2.1705677e-01	[ 3.1724284e-01]	 1.8782377e-01


.. parsed-literal::

      18	 3.1058167e-01	 2.1547251e-01	 3.4062156e-01	 2.1880861e-01	[ 3.4461149e-01]	 2.0997667e-01


.. parsed-literal::

      19	 3.4926813e-01	 2.1272244e-01	 3.7940952e-01	 2.1586201e-01	[ 3.8396269e-01]	 2.1561527e-01


.. parsed-literal::

      20	 4.2995553e-01	 2.0869346e-01	 4.6210242e-01	 2.1206690e-01	[ 4.6904164e-01]	 2.1733141e-01


.. parsed-literal::

      21	 5.4112938e-01	 2.0757276e-01	 5.7874328e-01	 2.1308496e-01	[ 6.1821463e-01]	 2.1347237e-01


.. parsed-literal::

      22	 5.8810491e-01	 2.0395208e-01	 6.2562860e-01	 2.0863496e-01	[ 6.6723863e-01]	 2.0970845e-01


.. parsed-literal::

      23	 6.3736223e-01	 1.9776404e-01	 6.7511501e-01	 2.0056693e-01	[ 7.2049369e-01]	 2.1631908e-01
      24	 6.8194607e-01	 1.9981151e-01	 7.1909679e-01	 1.9755367e-01	[ 7.7461153e-01]	 1.8565083e-01


.. parsed-literal::

      25	 7.1607570e-01	 2.0004962e-01	 7.5138433e-01	 1.9864343e-01	[ 7.9146635e-01]	 2.1867657e-01
      26	 7.4345920e-01	 1.9211064e-01	 7.8138161e-01	 1.8885476e-01	[ 8.2856419e-01]	 1.8645167e-01


.. parsed-literal::

      27	 7.7447418e-01	 1.9189558e-01	 8.1240731e-01	 1.8765158e-01	[ 8.5990222e-01]	 2.1022224e-01
      28	 8.0448508e-01	 1.9002974e-01	 8.4294772e-01	 1.8605853e-01	[ 8.8341268e-01]	 2.0061302e-01


.. parsed-literal::

      29	 8.3762641e-01	 1.8551114e-01	 8.7700246e-01	 1.8350772e-01	[ 9.0199087e-01]	 1.9549441e-01


.. parsed-literal::

      30	 8.6638715e-01	 1.8206875e-01	 9.0595354e-01	 1.8052421e-01	[ 9.2596732e-01]	 2.0408368e-01


.. parsed-literal::

      31	 8.9405535e-01	 1.7886675e-01	 9.3599904e-01	 1.7675615e-01	[ 9.6092372e-01]	 2.0694566e-01
      32	 9.2565943e-01	 1.7852699e-01	 9.6795382e-01	 1.7625680e-01	[ 9.9063207e-01]	 1.8849444e-01


.. parsed-literal::

      33	 9.4344018e-01	 1.7689572e-01	 9.8589559e-01	 1.7479638e-01	[ 1.0005368e+00]	 2.0150757e-01


.. parsed-literal::

      34	 9.6239091e-01	 1.7557548e-01	 1.0056382e+00	 1.7430349e-01	[ 1.0140421e+00]	 2.0714188e-01


.. parsed-literal::

      35	 9.8000845e-01	 1.7431148e-01	 1.0238895e+00	 1.7254100e-01	[ 1.0308981e+00]	 2.0687532e-01


.. parsed-literal::

      36	 9.9428382e-01	 1.7382304e-01	 1.0386812e+00	 1.7256250e-01	[ 1.0467329e+00]	 2.0595431e-01


.. parsed-literal::

      37	 1.0106664e+00	 1.7201796e-01	 1.0559521e+00	 1.7188557e-01	[ 1.0632538e+00]	 2.0739532e-01


.. parsed-literal::

      38	 1.0219367e+00	 1.7295532e-01	 1.0673774e+00	 1.7283609e-01	[ 1.0735642e+00]	 2.0785642e-01


.. parsed-literal::

      39	 1.0308724e+00	 1.7193078e-01	 1.0760863e+00	 1.7205185e-01	[ 1.0818039e+00]	 2.1472168e-01
      40	 1.0394704e+00	 1.7138231e-01	 1.0850322e+00	 1.7090022e-01	[ 1.0887017e+00]	 2.0000482e-01


.. parsed-literal::

      41	 1.0522495e+00	 1.7093276e-01	 1.0981872e+00	 1.6962972e-01	[ 1.0988171e+00]	 2.1297550e-01


.. parsed-literal::

      42	 1.0636083e+00	 1.7147807e-01	 1.1099790e+00	 1.6842764e-01	[ 1.1062724e+00]	 2.0422149e-01


.. parsed-literal::

      43	 1.0734959e+00	 1.7079778e-01	 1.1198150e+00	 1.6756782e-01	[ 1.1162070e+00]	 2.1471143e-01


.. parsed-literal::

      44	 1.0810991e+00	 1.6939641e-01	 1.1273539e+00	 1.6605555e-01	[ 1.1233775e+00]	 2.1730614e-01
      45	 1.0932749e+00	 1.6707466e-01	 1.1398897e+00	 1.6416789e-01	[ 1.1306875e+00]	 1.9271660e-01


.. parsed-literal::

      46	 1.1003436e+00	 1.6541185e-01	 1.1474541e+00	 1.6233696e-01	[ 1.1345559e+00]	 1.8793654e-01


.. parsed-literal::

      47	 1.1100549e+00	 1.6465739e-01	 1.1570563e+00	 1.6254034e-01	[ 1.1416419e+00]	 2.0919418e-01


.. parsed-literal::

      48	 1.1151684e+00	 1.6442419e-01	 1.1621906e+00	 1.6257689e-01	[ 1.1462224e+00]	 2.0480776e-01
      49	 1.1235010e+00	 1.6391614e-01	 1.1706764e+00	 1.6228442e-01	[ 1.1546711e+00]	 2.0342922e-01


.. parsed-literal::

      50	 1.1370354e+00	 1.6211247e-01	 1.1846708e+00	 1.6108875e-01	[ 1.1657116e+00]	 1.8545651e-01


.. parsed-literal::

      51	 1.1409158e+00	 1.6193345e-01	 1.1894209e+00	 1.6062549e-01	[ 1.1697553e+00]	 2.1234155e-01
      52	 1.1552309e+00	 1.5976217e-01	 1.2033760e+00	 1.5869765e-01	[ 1.1820128e+00]	 1.8944764e-01


.. parsed-literal::

      53	 1.1599478e+00	 1.5902137e-01	 1.2081267e+00	 1.5793634e-01	[ 1.1860507e+00]	 2.0882869e-01


.. parsed-literal::

      54	 1.1686454e+00	 1.5781834e-01	 1.2170749e+00	 1.5640929e-01	[ 1.1949875e+00]	 2.1357584e-01


.. parsed-literal::

      55	 1.1803509e+00	 1.5666450e-01	 1.2290859e+00	 1.5467407e-01	[ 1.2069658e+00]	 2.2030544e-01


.. parsed-literal::

      56	 1.1873846e+00	 1.5471361e-01	 1.2368809e+00	 1.5269578e-01	[ 1.2106973e+00]	 2.1323442e-01


.. parsed-literal::

      57	 1.1968694e+00	 1.5459053e-01	 1.2461059e+00	 1.5269483e-01	[ 1.2202627e+00]	 2.1061802e-01


.. parsed-literal::

      58	 1.2012961e+00	 1.5449454e-01	 1.2504762e+00	 1.5295707e-01	[ 1.2221584e+00]	 2.1089673e-01


.. parsed-literal::

      59	 1.2112436e+00	 1.5407587e-01	 1.2607662e+00	 1.5334502e-01	[ 1.2259977e+00]	 2.2070503e-01
      60	 1.2156667e+00	 1.5431613e-01	 1.2656136e+00	 1.5406265e-01	  1.2241347e+00 	 1.9123220e-01


.. parsed-literal::

      61	 1.2269651e+00	 1.5314920e-01	 1.2767697e+00	 1.5333014e-01	[ 1.2327440e+00]	 2.1473312e-01


.. parsed-literal::

      62	 1.2314990e+00	 1.5272254e-01	 1.2813165e+00	 1.5273721e-01	[ 1.2386395e+00]	 2.2387719e-01


.. parsed-literal::

      63	 1.2399802e+00	 1.5232046e-01	 1.2900821e+00	 1.5203421e-01	[ 1.2461109e+00]	 2.1862197e-01


.. parsed-literal::

      64	 1.2427996e+00	 1.5288951e-01	 1.2935943e+00	 1.5245133e-01	[ 1.2538268e+00]	 2.0303845e-01


.. parsed-literal::

      65	 1.2499179e+00	 1.5275211e-01	 1.3004298e+00	 1.5219174e-01	[ 1.2576787e+00]	 2.0932627e-01


.. parsed-literal::

      66	 1.2531359e+00	 1.5289381e-01	 1.3037331e+00	 1.5246707e-01	[ 1.2585860e+00]	 2.1772861e-01


.. parsed-literal::

      67	 1.2587039e+00	 1.5327128e-01	 1.3094639e+00	 1.5289397e-01	[ 1.2622851e+00]	 2.1198583e-01


.. parsed-literal::

      68	 1.2672750e+00	 1.5312706e-01	 1.3182367e+00	 1.5291824e-01	[ 1.2689489e+00]	 2.1056747e-01


.. parsed-literal::

      69	 1.2708029e+00	 1.5339845e-01	 1.3226976e+00	 1.5258058e-01	[ 1.2742160e+00]	 2.0152831e-01


.. parsed-literal::

      70	 1.2819308e+00	 1.5214672e-01	 1.3334025e+00	 1.5190091e-01	[ 1.2835224e+00]	 2.0804596e-01


.. parsed-literal::

      71	 1.2853117e+00	 1.5148390e-01	 1.3366994e+00	 1.5140904e-01	[ 1.2872174e+00]	 2.1463060e-01


.. parsed-literal::

      72	 1.2910673e+00	 1.5042737e-01	 1.3425410e+00	 1.5076960e-01	[ 1.2931436e+00]	 2.0269418e-01


.. parsed-literal::

      73	 1.2945427e+00	 1.4950023e-01	 1.3462269e+00	 1.5006425e-01	[ 1.2998680e+00]	 2.1592069e-01


.. parsed-literal::

      74	 1.3007930e+00	 1.4903669e-01	 1.3524782e+00	 1.5016956e-01	[ 1.3028488e+00]	 2.1240687e-01


.. parsed-literal::

      75	 1.3045810e+00	 1.4883263e-01	 1.3563219e+00	 1.5038832e-01	[ 1.3035067e+00]	 2.1928906e-01
      76	 1.3086246e+00	 1.4845760e-01	 1.3604499e+00	 1.5057824e-01	  1.3031032e+00 	 2.0644736e-01


.. parsed-literal::

      77	 1.3143602e+00	 1.4781787e-01	 1.3662326e+00	 1.5067303e-01	[ 1.3056198e+00]	 2.0434976e-01
      78	 1.3174347e+00	 1.4702492e-01	 1.3697428e+00	 1.5121010e-01	  1.2917973e+00 	 1.8789339e-01


.. parsed-literal::

      79	 1.3249278e+00	 1.4666883e-01	 1.3767754e+00	 1.5045206e-01	[ 1.3072546e+00]	 2.0666218e-01


.. parsed-literal::

      80	 1.3285062e+00	 1.4632833e-01	 1.3802843e+00	 1.4999909e-01	[ 1.3132520e+00]	 2.1321845e-01


.. parsed-literal::

      81	 1.3336853e+00	 1.4578960e-01	 1.3855285e+00	 1.4928104e-01	[ 1.3188766e+00]	 2.1448851e-01


.. parsed-literal::

      82	 1.3417528e+00	 1.4505833e-01	 1.3938576e+00	 1.4838283e-01	[ 1.3231987e+00]	 2.2090840e-01
      83	 1.3447377e+00	 1.4391582e-01	 1.3974873e+00	 1.4668765e-01	  1.3180009e+00 	 1.7796898e-01


.. parsed-literal::

      84	 1.3529212e+00	 1.4414033e-01	 1.4052646e+00	 1.4702240e-01	[ 1.3267718e+00]	 1.9696999e-01
      85	 1.3554924e+00	 1.4402999e-01	 1.4078564e+00	 1.4697871e-01	[ 1.3278039e+00]	 1.8595028e-01


.. parsed-literal::

      86	 1.3623880e+00	 1.4368938e-01	 1.4150470e+00	 1.4652994e-01	[ 1.3310375e+00]	 2.0914936e-01


.. parsed-literal::

      87	 1.3659490e+00	 1.4328751e-01	 1.4189310e+00	 1.4605871e-01	  1.3281058e+00 	 2.1075320e-01


.. parsed-literal::

      88	 1.3714112e+00	 1.4316923e-01	 1.4242969e+00	 1.4581068e-01	[ 1.3355346e+00]	 2.1460629e-01
      89	 1.3757687e+00	 1.4305184e-01	 1.4287225e+00	 1.4545571e-01	[ 1.3398463e+00]	 1.9583821e-01


.. parsed-literal::

      90	 1.3791135e+00	 1.4297855e-01	 1.4321184e+00	 1.4524547e-01	[ 1.3410147e+00]	 2.0557547e-01


.. parsed-literal::

      91	 1.3820550e+00	 1.4283958e-01	 1.4353978e+00	 1.4471068e-01	  1.3328669e+00 	 2.0412612e-01
      92	 1.3871095e+00	 1.4263663e-01	 1.4403247e+00	 1.4468497e-01	  1.3391168e+00 	 1.8643832e-01


.. parsed-literal::

      93	 1.3892188e+00	 1.4241196e-01	 1.4423981e+00	 1.4457556e-01	  1.3401241e+00 	 2.0194507e-01
      94	 1.3949397e+00	 1.4162610e-01	 1.4482247e+00	 1.4413118e-01	  1.3375228e+00 	 1.8308854e-01


.. parsed-literal::

      95	 1.3964318e+00	 1.4111511e-01	 1.4498328e+00	 1.4377965e-01	  1.3260499e+00 	 2.0980406e-01


.. parsed-literal::

      96	 1.4014706e+00	 1.4102555e-01	 1.4547477e+00	 1.4371765e-01	  1.3322966e+00 	 2.1423888e-01


.. parsed-literal::

      97	 1.4039954e+00	 1.4096702e-01	 1.4572724e+00	 1.4366646e-01	  1.3328378e+00 	 2.0870066e-01
      98	 1.4069879e+00	 1.4090124e-01	 1.4603009e+00	 1.4361871e-01	  1.3322849e+00 	 1.9787049e-01


.. parsed-literal::

      99	 1.4109694e+00	 1.4078293e-01	 1.4644006e+00	 1.4351261e-01	  1.3314568e+00 	 1.7528868e-01
     100	 1.4145953e+00	 1.4074377e-01	 1.4682597e+00	 1.4336143e-01	  1.3321164e+00 	 1.7659330e-01


.. parsed-literal::

     101	 1.4177520e+00	 1.4041192e-01	 1.4713284e+00	 1.4332759e-01	  1.3373547e+00 	 2.0557022e-01
     102	 1.4200180e+00	 1.4017522e-01	 1.4735467e+00	 1.4325434e-01	  1.3400333e+00 	 1.9600654e-01


.. parsed-literal::

     103	 1.4233542e+00	 1.3983270e-01	 1.4768918e+00	 1.4328518e-01	[ 1.3426676e+00]	 2.0435739e-01
     104	 1.4265654e+00	 1.3962126e-01	 1.4801485e+00	 1.4322320e-01	  1.3399198e+00 	 2.0416737e-01


.. parsed-literal::

     105	 1.4305623e+00	 1.3938644e-01	 1.4841081e+00	 1.4327019e-01	  1.3424367e+00 	 2.0118427e-01
     106	 1.4334193e+00	 1.3922848e-01	 1.4870173e+00	 1.4332201e-01	  1.3404526e+00 	 2.0358229e-01


.. parsed-literal::

     107	 1.4357919e+00	 1.3913742e-01	 1.4894748e+00	 1.4341232e-01	  1.3390849e+00 	 2.0908523e-01
     108	 1.4384765e+00	 1.3894429e-01	 1.4924641e+00	 1.4367773e-01	  1.3239787e+00 	 1.9863987e-01


.. parsed-literal::

     109	 1.4422554e+00	 1.3880034e-01	 1.4961966e+00	 1.4370366e-01	  1.3312736e+00 	 2.1273398e-01


.. parsed-literal::

     110	 1.4444964e+00	 1.3862743e-01	 1.4984612e+00	 1.4368743e-01	  1.3339001e+00 	 2.1799755e-01


.. parsed-literal::

     111	 1.4474285e+00	 1.3838234e-01	 1.5014593e+00	 1.4362850e-01	  1.3357341e+00 	 2.0563984e-01


.. parsed-literal::

     112	 1.4495179e+00	 1.3817428e-01	 1.5036334e+00	 1.4379691e-01	  1.3358024e+00 	 3.0362463e-01
     113	 1.4525878e+00	 1.3795799e-01	 1.5067820e+00	 1.4367052e-01	  1.3386239e+00 	 2.0243120e-01


.. parsed-literal::

     114	 1.4547907e+00	 1.3783190e-01	 1.5090355e+00	 1.4364562e-01	  1.3384935e+00 	 2.1050763e-01


.. parsed-literal::

     115	 1.4572339e+00	 1.3771317e-01	 1.5115851e+00	 1.4368645e-01	  1.3391036e+00 	 2.1041703e-01


.. parsed-literal::

     116	 1.4599028e+00	 1.3763556e-01	 1.5143297e+00	 1.4378796e-01	  1.3397101e+00 	 2.2822785e-01


.. parsed-literal::

     117	 1.4628625e+00	 1.3746892e-01	 1.5174021e+00	 1.4383007e-01	  1.3415555e+00 	 2.1567774e-01


.. parsed-literal::

     118	 1.4650099e+00	 1.3741138e-01	 1.5196069e+00	 1.4391226e-01	[ 1.3458149e+00]	 2.2094536e-01


.. parsed-literal::

     119	 1.4669139e+00	 1.3727999e-01	 1.5214284e+00	 1.4364492e-01	[ 1.3493518e+00]	 2.1507072e-01
     120	 1.4695564e+00	 1.3713113e-01	 1.5239824e+00	 1.4328031e-01	[ 1.3537041e+00]	 1.9967103e-01


.. parsed-literal::

     121	 1.4721962e+00	 1.3699652e-01	 1.5265811e+00	 1.4289850e-01	[ 1.3569855e+00]	 2.0579410e-01


.. parsed-literal::

     122	 1.4750081e+00	 1.3708971e-01	 1.5293615e+00	 1.4270630e-01	[ 1.3587269e+00]	 2.0446396e-01


.. parsed-literal::

     123	 1.4770598e+00	 1.3705953e-01	 1.5314633e+00	 1.4276472e-01	  1.3560667e+00 	 2.0583439e-01


.. parsed-literal::

     124	 1.4794817e+00	 1.3713518e-01	 1.5340049e+00	 1.4281845e-01	  1.3516528e+00 	 2.0540667e-01


.. parsed-literal::

     125	 1.4811780e+00	 1.3710605e-01	 1.5357465e+00	 1.4303183e-01	  1.3490455e+00 	 2.0709181e-01


.. parsed-literal::

     126	 1.4826023e+00	 1.3705507e-01	 1.5371565e+00	 1.4310599e-01	  1.3506260e+00 	 2.0513749e-01


.. parsed-literal::

     127	 1.4851111e+00	 1.3708142e-01	 1.5396524e+00	 1.4329702e-01	  1.3531465e+00 	 2.0887876e-01
     128	 1.4865121e+00	 1.3705940e-01	 1.5411641e+00	 1.4320155e-01	  1.3525560e+00 	 1.9139075e-01


.. parsed-literal::

     129	 1.4886287e+00	 1.3708052e-01	 1.5432095e+00	 1.4325276e-01	  1.3545200e+00 	 1.9423485e-01


.. parsed-literal::

     130	 1.4903764e+00	 1.3703614e-01	 1.5449523e+00	 1.4308806e-01	  1.3543295e+00 	 2.0894265e-01


.. parsed-literal::

     131	 1.4917608e+00	 1.3690320e-01	 1.5463607e+00	 1.4274587e-01	  1.3518314e+00 	 2.0556235e-01
     132	 1.4933587e+00	 1.3671507e-01	 1.5480687e+00	 1.4231058e-01	  1.3465320e+00 	 1.7244172e-01


.. parsed-literal::

     133	 1.4951334e+00	 1.3637487e-01	 1.5499137e+00	 1.4177638e-01	  1.3394087e+00 	 1.9655800e-01
     134	 1.4963826e+00	 1.3627039e-01	 1.5511788e+00	 1.4167493e-01	  1.3381053e+00 	 1.9818759e-01


.. parsed-literal::

     135	 1.4980997e+00	 1.3607130e-01	 1.5529747e+00	 1.4145970e-01	  1.3366166e+00 	 2.0118904e-01


.. parsed-literal::

     136	 1.4991794e+00	 1.3579360e-01	 1.5541888e+00	 1.4116076e-01	  1.3319994e+00 	 2.0779419e-01


.. parsed-literal::

     137	 1.5006630e+00	 1.3575666e-01	 1.5556188e+00	 1.4104282e-01	  1.3346826e+00 	 2.0224190e-01


.. parsed-literal::

     138	 1.5018855e+00	 1.3562607e-01	 1.5568374e+00	 1.4079249e-01	  1.3355775e+00 	 2.0997119e-01
     139	 1.5028773e+00	 1.3550015e-01	 1.5578260e+00	 1.4061305e-01	  1.3356540e+00 	 1.9766879e-01


.. parsed-literal::

     140	 1.5056158e+00	 1.3516878e-01	 1.5606079e+00	 1.4025054e-01	  1.3345796e+00 	 2.1320009e-01


.. parsed-literal::

     141	 1.5065498e+00	 1.3498146e-01	 1.5615609e+00	 1.4005326e-01	  1.3333578e+00 	 3.2940173e-01
     142	 1.5082032e+00	 1.3479618e-01	 1.5632625e+00	 1.3997699e-01	  1.3312860e+00 	 1.9646692e-01


.. parsed-literal::

     143	 1.5098999e+00	 1.3466782e-01	 1.5650121e+00	 1.3997632e-01	  1.3285121e+00 	 2.0395780e-01
     144	 1.5116886e+00	 1.3450353e-01	 1.5668925e+00	 1.4010319e-01	  1.3248240e+00 	 1.9920254e-01


.. parsed-literal::

     145	 1.5132248e+00	 1.3425138e-01	 1.5684755e+00	 1.3988604e-01	  1.3182126e+00 	 2.1015787e-01


.. parsed-literal::

     146	 1.5146351e+00	 1.3414538e-01	 1.5698741e+00	 1.3970620e-01	  1.3160730e+00 	 2.1484971e-01


.. parsed-literal::

     147	 1.5159031e+00	 1.3394455e-01	 1.5711254e+00	 1.3935396e-01	  1.3122573e+00 	 2.1363902e-01
     148	 1.5170328e+00	 1.3383468e-01	 1.5722570e+00	 1.3901965e-01	  1.3076102e+00 	 1.9903779e-01


.. parsed-literal::

     149	 1.5186197e+00	 1.3363551e-01	 1.5738800e+00	 1.3856662e-01	  1.2990831e+00 	 1.9614053e-01


.. parsed-literal::

     150	 1.5203504e+00	 1.3350499e-01	 1.5756675e+00	 1.3813256e-01	  1.2914364e+00 	 2.1634293e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.05 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3f9c62ea10>



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
    CPU times: user 2.15 s, sys: 42 ms, total: 2.19 s
    Wall time: 668 ms


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

