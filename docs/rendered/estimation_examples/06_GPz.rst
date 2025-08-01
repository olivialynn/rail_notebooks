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
       1	-3.5237972e-01	 3.2300129e-01	-3.4266816e-01	 3.1171395e-01	[-3.2118791e-01]	 4.6469951e-01


.. parsed-literal::

       2	-2.7920894e-01	 3.1183275e-01	-2.5518059e-01	 3.0162221e-01	[-2.2504705e-01]	 2.3128557e-01


.. parsed-literal::

       3	-2.3368486e-01	 2.9120087e-01	-1.9233338e-01	 2.8482730e-01	[-1.6586246e-01]	 2.8558683e-01


.. parsed-literal::

       4	-1.9309604e-01	 2.7291428e-01	-1.4466260e-01	 2.7098419e-01	[-1.2597652e-01]	 3.0543137e-01


.. parsed-literal::

       5	-1.4207534e-01	 2.5969500e-01	-1.1220529e-01	 2.5664196e-01	[-9.2365668e-02]	 2.0678282e-01
       6	-8.1895247e-02	 2.5527545e-01	-5.3529781e-02	 2.5169932e-01	[-3.6468345e-02]	 1.8608665e-01


.. parsed-literal::

       7	-5.9619134e-02	 2.5064042e-01	-3.6179052e-02	 2.4597634e-01	[-1.8226950e-02]	 2.2379327e-01


.. parsed-literal::

       8	-4.8103918e-02	 2.4861846e-01	-2.7516029e-02	 2.4382746e-01	[-6.9859769e-03]	 2.0847297e-01


.. parsed-literal::

       9	-3.3569381e-02	 2.4589783e-01	-1.6033082e-02	 2.4120957e-01	[ 4.2772898e-03]	 2.0934176e-01
      10	-2.3004427e-02	 2.4388870e-01	-7.3836697e-03	 2.4023640e-01	[ 9.9641398e-03]	 2.0123863e-01


.. parsed-literal::

      11	-1.6233554e-02	 2.4278941e-01	-1.9864996e-03	 2.3944350e-01	[ 1.4416975e-02]	 1.9712615e-01


.. parsed-literal::

      12	-1.2396876e-02	 2.4200082e-01	 1.6557282e-03	 2.3800725e-01	[ 1.9061858e-02]	 2.0843077e-01


.. parsed-literal::

      13	-8.0598909e-03	 2.4117709e-01	 5.8769748e-03	 2.3663902e-01	[ 2.5138644e-02]	 2.0894861e-01


.. parsed-literal::

      14	 6.4461414e-02	 2.3014606e-01	 8.3071084e-02	 2.2275324e-01	[ 1.1296895e-01]	 3.2520461e-01


.. parsed-literal::

      15	 9.8661352e-02	 2.2815584e-01	 1.1967649e-01	 2.2343937e-01	[ 1.3998007e-01]	 4.3756890e-01


.. parsed-literal::

      16	 1.4743710e-01	 2.2409526e-01	 1.6948592e-01	 2.1697275e-01	[ 1.9307309e-01]	 2.1025562e-01
      17	 2.2431432e-01	 2.1993847e-01	 2.5180078e-01	 2.1255701e-01	[ 2.8123159e-01]	 1.9971371e-01


.. parsed-literal::

      18	 2.7666561e-01	 2.1397529e-01	 3.0865166e-01	 2.0797964e-01	[ 3.4309440e-01]	 1.9797492e-01


.. parsed-literal::

      19	 3.4348936e-01	 2.0634561e-01	 3.7726364e-01	 2.0527761e-01	[ 3.9235702e-01]	 2.1475935e-01


.. parsed-literal::

      20	 3.9303146e-01	 2.0244960e-01	 4.2765393e-01	 2.0901754e-01	[ 4.2112459e-01]	 2.1003056e-01
      21	 4.5773164e-01	 1.9409893e-01	 4.9351910e-01	 2.0226797e-01	[ 4.8389446e-01]	 1.9703269e-01


.. parsed-literal::

      22	 5.6336447e-01	 1.8673312e-01	 6.0085785e-01	 1.9578355e-01	[ 5.7811347e-01]	 2.0661139e-01


.. parsed-literal::

      23	 6.3073254e-01	 1.8303714e-01	 6.6978595e-01	 1.9118219e-01	[ 6.7665976e-01]	 2.0297956e-01


.. parsed-literal::

      24	 6.8198921e-01	 1.7946626e-01	 7.2142409e-01	 1.9008546e-01	[ 7.1962822e-01]	 2.0987916e-01


.. parsed-literal::

      25	 7.1468222e-01	 1.7783897e-01	 7.5475712e-01	 1.8603786e-01	[ 7.5290217e-01]	 2.1228194e-01


.. parsed-literal::

      26	 7.3613051e-01	 1.7470990e-01	 7.7393879e-01	 1.8132173e-01	[ 7.6697741e-01]	 2.1312332e-01


.. parsed-literal::

      27	 7.7228796e-01	 1.7511946e-01	 8.1162419e-01	 1.7980924e-01	[ 8.0411477e-01]	 2.0522141e-01


.. parsed-literal::

      28	 8.0149282e-01	 1.7651227e-01	 8.4279139e-01	 1.8030979e-01	[ 8.3566240e-01]	 2.1340013e-01


.. parsed-literal::

      29	 8.3871656e-01	 1.7346072e-01	 8.7999946e-01	 1.7498764e-01	[ 8.7812261e-01]	 2.0844007e-01
      30	 8.7370604e-01	 1.7259764e-01	 9.1483222e-01	 1.7308082e-01	[ 9.1407341e-01]	 1.7532396e-01


.. parsed-literal::

      31	 9.0122434e-01	 1.6866228e-01	 9.4194100e-01	 1.7195260e-01	[ 9.4110405e-01]	 1.9772625e-01


.. parsed-literal::

      32	 9.1840454e-01	 1.6762777e-01	 9.5972524e-01	 1.7226313e-01	[ 9.5312215e-01]	 2.1125484e-01


.. parsed-literal::

      33	 9.4683667e-01	 1.6732393e-01	 9.9024092e-01	 1.7408779e-01	[ 9.6841431e-01]	 2.3310280e-01


.. parsed-literal::

      34	 9.5957362e-01	 1.6354987e-01	 1.0042782e+00	 1.7273497e-01	[ 9.6971735e-01]	 2.0733380e-01


.. parsed-literal::

      35	 9.7806282e-01	 1.6322770e-01	 1.0223987e+00	 1.7274624e-01	[ 9.8819610e-01]	 2.0756292e-01


.. parsed-literal::

      36	 9.9143660e-01	 1.6212669e-01	 1.0357564e+00	 1.7113192e-01	[ 1.0015226e+00]	 2.0873427e-01


.. parsed-literal::

      37	 1.0088119e+00	 1.6077985e-01	 1.0533893e+00	 1.6903386e-01	[ 1.0176182e+00]	 2.1754909e-01


.. parsed-literal::

      38	 1.0349013e+00	 1.5872534e-01	 1.0805725e+00	 1.6669390e-01	[ 1.0396463e+00]	 2.1244621e-01


.. parsed-literal::

      39	 1.0482294e+00	 1.5798371e-01	 1.0944234e+00	 1.6578366e-01	[ 1.0517606e+00]	 3.3094835e-01


.. parsed-literal::

      40	 1.0613796e+00	 1.5552329e-01	 1.1080513e+00	 1.6433723e-01	[ 1.0651534e+00]	 2.1324635e-01


.. parsed-literal::

      41	 1.0774499e+00	 1.5264495e-01	 1.1246522e+00	 1.6171977e-01	[ 1.0781831e+00]	 2.1625566e-01


.. parsed-literal::

      42	 1.0925013e+00	 1.5057789e-01	 1.1397411e+00	 1.5870555e-01	[ 1.0985404e+00]	 2.2029972e-01


.. parsed-literal::

      43	 1.1069820e+00	 1.4690119e-01	 1.1541816e+00	 1.5518879e-01	[ 1.1085142e+00]	 2.0920396e-01


.. parsed-literal::

      44	 1.1193815e+00	 1.4573811e-01	 1.1663342e+00	 1.5395298e-01	[ 1.1219436e+00]	 2.0412087e-01


.. parsed-literal::

      45	 1.1305608e+00	 1.4417223e-01	 1.1776275e+00	 1.5214852e-01	[ 1.1318144e+00]	 2.1396804e-01


.. parsed-literal::

      46	 1.1425141e+00	 1.4231223e-01	 1.1896895e+00	 1.5213485e-01	[ 1.1340341e+00]	 2.0526528e-01
      47	 1.1541121e+00	 1.4046417e-01	 1.2015726e+00	 1.5063744e-01	[ 1.1433868e+00]	 1.8657613e-01


.. parsed-literal::

      48	 1.1637533e+00	 1.4002900e-01	 1.2112088e+00	 1.5110026e-01	[ 1.1492572e+00]	 2.1565795e-01


.. parsed-literal::

      49	 1.1812426e+00	 1.3916847e-01	 1.2292547e+00	 1.5150418e-01	[ 1.1679078e+00]	 2.0841980e-01
      50	 1.1923307e+00	 1.3867317e-01	 1.2404497e+00	 1.5145018e-01	[ 1.1819477e+00]	 1.8611193e-01


.. parsed-literal::

      51	 1.2027590e+00	 1.3778730e-01	 1.2507256e+00	 1.4966600e-01	[ 1.1981145e+00]	 1.9226742e-01
      52	 1.2166434e+00	 1.3618536e-01	 1.2648573e+00	 1.4717487e-01	[ 1.2116570e+00]	 2.0239758e-01


.. parsed-literal::

      53	 1.2279219e+00	 1.3505056e-01	 1.2762855e+00	 1.4606710e-01	[ 1.2180962e+00]	 2.1378779e-01


.. parsed-literal::

      54	 1.2356030e+00	 1.3471721e-01	 1.2846726e+00	 1.4618749e-01	  1.2064942e+00 	 2.1878839e-01


.. parsed-literal::

      55	 1.2470812e+00	 1.3346613e-01	 1.2959917e+00	 1.4562906e-01	  1.2155258e+00 	 2.0586991e-01


.. parsed-literal::

      56	 1.2531625e+00	 1.3281521e-01	 1.3021585e+00	 1.4544585e-01	[ 1.2247663e+00]	 2.0885897e-01


.. parsed-literal::

      57	 1.2643993e+00	 1.3168559e-01	 1.3136700e+00	 1.4536442e-01	[ 1.2263215e+00]	 2.1415567e-01


.. parsed-literal::

      58	 1.2776090e+00	 1.3133565e-01	 1.3270917e+00	 1.4695689e-01	[ 1.2266733e+00]	 2.0544982e-01


.. parsed-literal::

      59	 1.2877507e+00	 1.3106364e-01	 1.3373928e+00	 1.4730035e-01	  1.2181764e+00 	 2.1008968e-01


.. parsed-literal::

      60	 1.2942863e+00	 1.3053837e-01	 1.3438667e+00	 1.4712891e-01	  1.2204595e+00 	 2.0227671e-01


.. parsed-literal::

      61	 1.3020909e+00	 1.2992141e-01	 1.3518154e+00	 1.4709274e-01	  1.2187437e+00 	 2.1274686e-01


.. parsed-literal::

      62	 1.3101519e+00	 1.2943139e-01	 1.3599196e+00	 1.4824776e-01	  1.2146347e+00 	 2.0740891e-01
      63	 1.3176789e+00	 1.2891042e-01	 1.3675415e+00	 1.4884774e-01	  1.2026222e+00 	 1.9735956e-01


.. parsed-literal::

      64	 1.3238578e+00	 1.2869137e-01	 1.3737605e+00	 1.4914382e-01	  1.1980291e+00 	 2.0454335e-01
      65	 1.3322574e+00	 1.2822959e-01	 1.3825920e+00	 1.4951299e-01	  1.1818118e+00 	 1.9285607e-01


.. parsed-literal::

      66	 1.3386150e+00	 1.2754956e-01	 1.3893982e+00	 1.4914095e-01	  1.1686347e+00 	 2.0376420e-01
      67	 1.3468911e+00	 1.2733410e-01	 1.3975600e+00	 1.4911160e-01	  1.1618409e+00 	 1.8579388e-01


.. parsed-literal::

      68	 1.3523150e+00	 1.2706065e-01	 1.4028980e+00	 1.4833985e-01	  1.1725058e+00 	 2.0861864e-01
      69	 1.3573124e+00	 1.2685103e-01	 1.4080182e+00	 1.4775490e-01	  1.1727372e+00 	 1.9423842e-01


.. parsed-literal::

      70	 1.3662656e+00	 1.2643456e-01	 1.4169181e+00	 1.4602172e-01	  1.1833950e+00 	 1.9565392e-01
      71	 1.3727033e+00	 1.2664761e-01	 1.4235313e+00	 1.4501036e-01	  1.1783617e+00 	 2.0110774e-01


.. parsed-literal::

      72	 1.3774403e+00	 1.2649521e-01	 1.4281972e+00	 1.4493223e-01	  1.1827385e+00 	 1.9663191e-01


.. parsed-literal::

      73	 1.3845591e+00	 1.2644525e-01	 1.4354569e+00	 1.4482740e-01	  1.1790066e+00 	 2.1705270e-01


.. parsed-literal::

      74	 1.3911097e+00	 1.2621270e-01	 1.4420542e+00	 1.4533230e-01	  1.1760425e+00 	 2.0741677e-01


.. parsed-literal::

      75	 1.3978745e+00	 1.2630182e-01	 1.4488153e+00	 1.4541222e-01	  1.1838979e+00 	 2.0926785e-01


.. parsed-literal::

      76	 1.4042642e+00	 1.2568552e-01	 1.4553704e+00	 1.4480778e-01	  1.1868336e+00 	 2.1802187e-01


.. parsed-literal::

      77	 1.4103571e+00	 1.2528658e-01	 1.4616436e+00	 1.4466957e-01	  1.1841979e+00 	 2.1204686e-01


.. parsed-literal::

      78	 1.4150858e+00	 1.2482958e-01	 1.4665780e+00	 1.4446144e-01	  1.1767750e+00 	 2.0840597e-01


.. parsed-literal::

      79	 1.4194981e+00	 1.2463466e-01	 1.4710051e+00	 1.4452264e-01	  1.1724708e+00 	 2.1605110e-01
      80	 1.4242295e+00	 1.2467881e-01	 1.4758430e+00	 1.4461332e-01	  1.1640762e+00 	 2.0021033e-01


.. parsed-literal::

      81	 1.4291821e+00	 1.2429643e-01	 1.4808872e+00	 1.4488539e-01	  1.1493562e+00 	 1.8680644e-01


.. parsed-literal::

      82	 1.4336009e+00	 1.2387252e-01	 1.4854890e+00	 1.4494033e-01	  1.1390018e+00 	 2.0444417e-01


.. parsed-literal::

      83	 1.4372782e+00	 1.2362891e-01	 1.4891842e+00	 1.4472634e-01	  1.1322610e+00 	 2.0549417e-01


.. parsed-literal::

      84	 1.4411195e+00	 1.2334923e-01	 1.4931452e+00	 1.4439764e-01	  1.1254925e+00 	 2.0925903e-01


.. parsed-literal::

      85	 1.4460538e+00	 1.2320693e-01	 1.4981683e+00	 1.4425815e-01	  1.1188600e+00 	 2.0826197e-01
      86	 1.4502949e+00	 1.2312411e-01	 1.5025672e+00	 1.4451875e-01	  1.1228420e+00 	 1.7729640e-01


.. parsed-literal::

      87	 1.4555831e+00	 1.2313778e-01	 1.5076745e+00	 1.4448758e-01	  1.1349260e+00 	 1.9665742e-01


.. parsed-literal::

      88	 1.4587897e+00	 1.2323005e-01	 1.5108452e+00	 1.4453287e-01	  1.1452549e+00 	 2.0707774e-01


.. parsed-literal::

      89	 1.4621114e+00	 1.2365031e-01	 1.5142363e+00	 1.4464083e-01	  1.1639490e+00 	 2.1168709e-01
      90	 1.4655533e+00	 1.2380886e-01	 1.5177529e+00	 1.4446148e-01	  1.1769545e+00 	 1.8861127e-01


.. parsed-literal::

      91	 1.4682699e+00	 1.2369015e-01	 1.5205085e+00	 1.4422579e-01	  1.1819989e+00 	 2.1926451e-01


.. parsed-literal::

      92	 1.4720477e+00	 1.2347173e-01	 1.5244304e+00	 1.4393740e-01	  1.1902284e+00 	 2.1747684e-01


.. parsed-literal::

      93	 1.4749154e+00	 1.2324321e-01	 1.5272925e+00	 1.4385332e-01	  1.1877352e+00 	 2.2025824e-01
      94	 1.4771673e+00	 1.2315793e-01	 1.5294893e+00	 1.4393653e-01	  1.1881991e+00 	 2.0267057e-01


.. parsed-literal::

      95	 1.4819660e+00	 1.2294016e-01	 1.5343665e+00	 1.4427060e-01	  1.1828003e+00 	 2.0423603e-01


.. parsed-literal::

      96	 1.4840802e+00	 1.2280136e-01	 1.5365236e+00	 1.4435565e-01	  1.1697827e+00 	 2.1715641e-01


.. parsed-literal::

      97	 1.4866563e+00	 1.2257711e-01	 1.5391651e+00	 1.4424106e-01	  1.1628319e+00 	 2.0932245e-01


.. parsed-literal::

      98	 1.4888813e+00	 1.2234807e-01	 1.5414847e+00	 1.4402977e-01	  1.1513488e+00 	 2.1373153e-01


.. parsed-literal::

      99	 1.4915648e+00	 1.2204063e-01	 1.5442823e+00	 1.4364582e-01	  1.1260029e+00 	 2.0925832e-01


.. parsed-literal::

     100	 1.4941768e+00	 1.2167672e-01	 1.5471202e+00	 1.4316333e-01	  1.0884024e+00 	 2.0783019e-01
     101	 1.4970522e+00	 1.2150958e-01	 1.5498906e+00	 1.4309807e-01	  1.0776092e+00 	 1.9915700e-01


.. parsed-literal::

     102	 1.4995484e+00	 1.2144035e-01	 1.5523087e+00	 1.4311666e-01	  1.0704635e+00 	 2.0860505e-01


.. parsed-literal::

     103	 1.5017666e+00	 1.2122922e-01	 1.5544996e+00	 1.4290286e-01	  1.0589875e+00 	 2.0944238e-01
     104	 1.5048119e+00	 1.2105403e-01	 1.5575217e+00	 1.4274996e-01	  1.0467849e+00 	 1.7224574e-01


.. parsed-literal::

     105	 1.5077400e+00	 1.2083882e-01	 1.5604673e+00	 1.4255835e-01	  1.0400074e+00 	 2.1130466e-01


.. parsed-literal::

     106	 1.5112693e+00	 1.2058067e-01	 1.5641241e+00	 1.4240184e-01	  1.0175134e+00 	 2.1716857e-01


.. parsed-literal::

     107	 1.5132452e+00	 1.2048192e-01	 1.5662686e+00	 1.4253764e-01	  1.0078005e+00 	 2.0498919e-01


.. parsed-literal::

     108	 1.5156458e+00	 1.2040675e-01	 1.5686031e+00	 1.4247290e-01	  1.0079792e+00 	 2.0904231e-01


.. parsed-literal::

     109	 1.5176615e+00	 1.2031233e-01	 1.5706195e+00	 1.4238803e-01	  9.9478911e-01 	 2.0936966e-01


.. parsed-literal::

     110	 1.5197596e+00	 1.2017294e-01	 1.5727289e+00	 1.4234730e-01	  9.9162468e-01 	 2.1520925e-01


.. parsed-literal::

     111	 1.5212047e+00	 1.1994481e-01	 1.5742668e+00	 1.4231181e-01	  9.8751028e-01 	 2.1851468e-01


.. parsed-literal::

     112	 1.5234024e+00	 1.1987866e-01	 1.5764026e+00	 1.4233186e-01	  9.9652721e-01 	 2.0859933e-01


.. parsed-literal::

     113	 1.5252276e+00	 1.1976815e-01	 1.5782383e+00	 1.4235354e-01	  1.0029410e+00 	 2.0709157e-01


.. parsed-literal::

     114	 1.5274185e+00	 1.1959274e-01	 1.5805158e+00	 1.4237707e-01	  1.0024227e+00 	 2.0270896e-01
     115	 1.5303541e+00	 1.1925914e-01	 1.5836530e+00	 1.4218663e-01	  9.9142805e-01 	 1.9910789e-01


.. parsed-literal::

     116	 1.5321742e+00	 1.1918636e-01	 1.5855716e+00	 1.4237276e-01	  9.8818517e-01 	 3.1166792e-01
     117	 1.5338080e+00	 1.1902711e-01	 1.5872344e+00	 1.4220645e-01	  9.8083621e-01 	 1.9761086e-01


.. parsed-literal::

     118	 1.5354969e+00	 1.1894879e-01	 1.5889683e+00	 1.4205432e-01	  9.7637375e-01 	 2.1334100e-01


.. parsed-literal::

     119	 1.5370597e+00	 1.1898646e-01	 1.5905146e+00	 1.4201762e-01	  9.7253944e-01 	 2.0619988e-01
     120	 1.5385130e+00	 1.1903430e-01	 1.5919683e+00	 1.4197651e-01	  9.7497662e-01 	 1.9097948e-01


.. parsed-literal::

     121	 1.5401031e+00	 1.1904648e-01	 1.5936065e+00	 1.4194827e-01	  9.7789727e-01 	 2.0114303e-01


.. parsed-literal::

     122	 1.5417086e+00	 1.1900218e-01	 1.5952891e+00	 1.4192713e-01	  9.7378395e-01 	 2.1248078e-01
     123	 1.5439729e+00	 1.1891874e-01	 1.5977174e+00	 1.4189643e-01	  9.6803374e-01 	 1.8325377e-01


.. parsed-literal::

     124	 1.5461216e+00	 1.1875683e-01	 1.5999794e+00	 1.4188690e-01	  9.5087536e-01 	 2.0999670e-01
     125	 1.5481342e+00	 1.1859292e-01	 1.6020077e+00	 1.4182599e-01	  9.3811853e-01 	 1.7196298e-01


.. parsed-literal::

     126	 1.5498469e+00	 1.1849228e-01	 1.6037351e+00	 1.4180916e-01	  9.2813535e-01 	 2.0070457e-01


.. parsed-literal::

     127	 1.5516282e+00	 1.1829982e-01	 1.6055363e+00	 1.4186593e-01	  9.2073562e-01 	 2.1513486e-01
     128	 1.5533339e+00	 1.1826400e-01	 1.6072333e+00	 1.4220756e-01	  9.1683739e-01 	 1.9717932e-01


.. parsed-literal::

     129	 1.5549033e+00	 1.1822350e-01	 1.6087629e+00	 1.4234424e-01	  9.1826163e-01 	 2.0650935e-01


.. parsed-literal::

     130	 1.5562912e+00	 1.1812783e-01	 1.6101500e+00	 1.4249361e-01	  9.1975349e-01 	 2.1617508e-01
     131	 1.5576745e+00	 1.1787502e-01	 1.6116176e+00	 1.4276052e-01	  9.1860338e-01 	 1.9935465e-01


.. parsed-literal::

     132	 1.5592239e+00	 1.1774950e-01	 1.6131417e+00	 1.4281013e-01	  9.1340583e-01 	 1.8372822e-01
     133	 1.5603678e+00	 1.1760419e-01	 1.6143049e+00	 1.4284922e-01	  9.0553345e-01 	 1.9095874e-01


.. parsed-literal::

     134	 1.5616005e+00	 1.1745395e-01	 1.6155520e+00	 1.4285365e-01	  9.0054886e-01 	 1.6998935e-01
     135	 1.5631928e+00	 1.1718209e-01	 1.6172277e+00	 1.4286869e-01	  8.9121306e-01 	 2.0053339e-01


.. parsed-literal::

     136	 1.5648968e+00	 1.1713621e-01	 1.6189088e+00	 1.4271666e-01	  8.9344834e-01 	 1.8686676e-01


.. parsed-literal::

     137	 1.5659376e+00	 1.1715387e-01	 1.6199350e+00	 1.4264721e-01	  8.9610257e-01 	 2.0995283e-01


.. parsed-literal::

     138	 1.5674520e+00	 1.1722659e-01	 1.6214823e+00	 1.4255255e-01	  8.9813483e-01 	 2.0957804e-01
     139	 1.5688409e+00	 1.1727488e-01	 1.6229835e+00	 1.4245211e-01	  8.9282637e-01 	 1.9979024e-01


.. parsed-literal::

     140	 1.5703431e+00	 1.1735101e-01	 1.6245051e+00	 1.4250817e-01	  8.9033524e-01 	 2.0752192e-01


.. parsed-literal::

     141	 1.5716349e+00	 1.1735659e-01	 1.6258436e+00	 1.4255088e-01	  8.8287220e-01 	 2.1411633e-01


.. parsed-literal::

     142	 1.5726898e+00	 1.1730033e-01	 1.6269379e+00	 1.4260329e-01	  8.7688872e-01 	 2.1164060e-01


.. parsed-literal::

     143	 1.5743038e+00	 1.1710678e-01	 1.6286633e+00	 1.4275066e-01	  8.6228739e-01 	 2.0625567e-01
     144	 1.5756312e+00	 1.1689299e-01	 1.6300127e+00	 1.4290289e-01	  8.5303934e-01 	 1.9542480e-01


.. parsed-literal::

     145	 1.5766445e+00	 1.1683027e-01	 1.6309869e+00	 1.4290990e-01	  8.5543310e-01 	 2.1558833e-01
     146	 1.5776118e+00	 1.1669277e-01	 1.6319227e+00	 1.4294255e-01	  8.5464804e-01 	 1.8650365e-01


.. parsed-literal::

     147	 1.5785741e+00	 1.1660250e-01	 1.6328803e+00	 1.4296057e-01	  8.4992089e-01 	 1.9823360e-01


.. parsed-literal::

     148	 1.5800143e+00	 1.1639582e-01	 1.6343671e+00	 1.4301731e-01	  8.2263558e-01 	 2.0964313e-01


.. parsed-literal::

     149	 1.5814476e+00	 1.1639019e-01	 1.6358127e+00	 1.4296391e-01	  8.1258497e-01 	 2.1198750e-01
     150	 1.5824019e+00	 1.1643543e-01	 1.6367589e+00	 1.4286522e-01	  8.0454645e-01 	 1.8145752e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.03 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1a28b946d0>



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
    CPU times: user 2.04 s, sys: 52.9 ms, total: 2.1 s
    Wall time: 630 ms


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

