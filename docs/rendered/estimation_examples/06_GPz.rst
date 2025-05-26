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
       1	-3.4095319e-01	 3.1965261e-01	-3.3133150e-01	 3.2402656e-01	[-3.3993969e-01]	 4.5955276e-01


.. parsed-literal::

       2	-2.6986306e-01	 3.0904367e-01	-2.4559443e-01	 3.1268790e-01	[-2.5719879e-01]	 2.2663236e-01


.. parsed-literal::

       3	-2.2232016e-01	 2.8599696e-01	-1.7674902e-01	 2.9218635e-01	[-2.0385836e-01]	 2.8636765e-01


.. parsed-literal::

       4	-1.9377416e-01	 2.6260974e-01	-1.5181871e-01	 2.7485811e-01	 -2.2203219e-01 	 2.0960021e-01


.. parsed-literal::

       5	-9.8269169e-02	 2.5616066e-01	-6.3999204e-02	 2.6494861e-01	[-1.0393302e-01]	 2.1013880e-01


.. parsed-literal::

       6	-6.6640840e-02	 2.5030909e-01	-3.5612606e-02	 2.5880542e-01	[-6.6172886e-02]	 2.1464086e-01


.. parsed-literal::

       7	-4.8930349e-02	 2.4786081e-01	-2.4591169e-02	 2.5601751e-01	[-5.5789203e-02]	 2.1127224e-01


.. parsed-literal::

       8	-3.5077207e-02	 2.4555011e-01	-1.4778148e-02	 2.5331884e-01	[-4.5707135e-02]	 2.2135091e-01


.. parsed-literal::

       9	-1.9323942e-02	 2.4251969e-01	-1.8843336e-03	 2.5055704e-01	[-3.3691563e-02]	 2.2009802e-01
      10	-8.7229420e-03	 2.4040265e-01	 6.7774118e-03	 2.4907395e-01	[-2.9513953e-02]	 2.0568466e-01


.. parsed-literal::

      11	-3.9542220e-03	 2.3980340e-01	 1.0631132e-02	 2.4896180e-01	[-2.5640320e-02]	 2.0087123e-01


.. parsed-literal::

      12	-1.1828284e-03	 2.3919211e-01	 1.3144579e-02	 2.4850742e-01	[-2.3250741e-02]	 2.3167038e-01


.. parsed-literal::

      13	 2.2213653e-03	 2.3860142e-01	 1.6348986e-02	 2.4800350e-01	[-2.1233704e-02]	 2.0467806e-01
      14	 8.4371059e-03	 2.3724872e-01	 2.3576645e-02	 2.4670307e-01	[-1.3711879e-02]	 1.9273138e-01


.. parsed-literal::

      15	 9.2001917e-02	 2.2567595e-01	 1.1099239e-01	 2.3151960e-01	[ 8.8142233e-02]	 1.8412614e-01
      16	 1.5277742e-01	 2.2238903e-01	 1.7647054e-01	 2.2679243e-01	[ 1.5671786e-01]	 1.8186355e-01


.. parsed-literal::

      17	 2.3957166e-01	 2.2013941e-01	 2.6844223e-01	 2.2563666e-01	[ 2.3639104e-01]	 2.2127008e-01
      18	 3.2326421e-01	 2.1630443e-01	 3.5541920e-01	 2.2651356e-01	[ 3.0248801e-01]	 1.7776346e-01


.. parsed-literal::

      19	 3.6181199e-01	 2.1430300e-01	 3.9527523e-01	 2.2430081e-01	[ 3.1728591e-01]	 2.1471667e-01
      20	 4.0010154e-01	 2.1075107e-01	 4.3364343e-01	 2.2314786e-01	[ 3.5216998e-01]	 1.9377589e-01


.. parsed-literal::

      21	 4.8594292e-01	 2.0515688e-01	 5.2042095e-01	 2.1694306e-01	[ 4.2610514e-01]	 2.0792866e-01
      22	 5.7618012e-01	 2.0208682e-01	 6.1270832e-01	 2.1193403e-01	[ 5.0860159e-01]	 1.8764472e-01


.. parsed-literal::

      23	 6.1999152e-01	 1.9885411e-01	 6.6128370e-01	 2.0956974e-01	[ 5.2277025e-01]	 2.0437646e-01
      24	 6.6808595e-01	 1.9552023e-01	 7.0643924e-01	 2.0563835e-01	[ 5.8965554e-01]	 1.9826388e-01


.. parsed-literal::

      25	 6.9359023e-01	 1.9527910e-01	 7.3172506e-01	 2.0681489e-01	[ 6.1482826e-01]	 1.9324946e-01


.. parsed-literal::

      26	 7.2838972e-01	 2.0387324e-01	 7.6603795e-01	 2.2051276e-01	[ 6.5398880e-01]	 2.0984006e-01
      27	 7.7368228e-01	 2.0115191e-01	 8.1167188e-01	 2.1240391e-01	[ 7.1160558e-01]	 1.8975711e-01


.. parsed-literal::

      28	 8.0511744e-01	 2.0129782e-01	 8.4453275e-01	 2.1199093e-01	[ 7.4772805e-01]	 1.8864202e-01
      29	 8.3352868e-01	 2.0272417e-01	 8.7327481e-01	 2.1278223e-01	[ 8.0023872e-01]	 2.0045686e-01


.. parsed-literal::

      30	 8.5810389e-01	 2.0615263e-01	 8.9935714e-01	 2.1478553e-01	[ 8.1807533e-01]	 2.1163130e-01
      31	 8.7733145e-01	 2.0427170e-01	 9.1938841e-01	 2.1267240e-01	[ 8.3185885e-01]	 1.7688704e-01


.. parsed-literal::

      32	 8.9978689e-01	 2.0341037e-01	 9.4236130e-01	 2.1154338e-01	[ 8.5406650e-01]	 1.9079494e-01


.. parsed-literal::

      33	 9.2668523e-01	 1.9699606e-01	 9.7098828e-01	 2.0689425e-01	[ 8.6842137e-01]	 2.0949125e-01


.. parsed-literal::

      34	 9.6055719e-01	 1.9419471e-01	 1.0050689e+00	 2.0431931e-01	[ 9.0467220e-01]	 2.1769118e-01
      35	 9.8089706e-01	 1.9305687e-01	 1.0260367e+00	 2.0378868e-01	[ 9.2335422e-01]	 1.9993186e-01


.. parsed-literal::

      36	 9.9399182e-01	 1.9260589e-01	 1.0389712e+00	 2.0332952e-01	[ 9.3711279e-01]	 2.0846534e-01


.. parsed-literal::

      37	 1.0048726e+00	 1.9251793e-01	 1.0498704e+00	 2.0293443e-01	[ 9.5207999e-01]	 2.0284081e-01


.. parsed-literal::

      38	 1.0176284e+00	 1.9128524e-01	 1.0632153e+00	 2.0192519e-01	[ 9.6437634e-01]	 2.0788717e-01


.. parsed-literal::

      39	 1.0321721e+00	 1.8988445e-01	 1.0786751e+00	 2.0128986e-01	[ 9.8006392e-01]	 2.1001697e-01


.. parsed-literal::

      40	 1.0435924e+00	 1.8548191e-01	 1.0911343e+00	 1.9762138e-01	[ 9.8865414e-01]	 2.0801711e-01


.. parsed-literal::

      41	 1.0546038e+00	 1.8331975e-01	 1.1019851e+00	 1.9539330e-01	[ 9.9765267e-01]	 2.1855497e-01


.. parsed-literal::

      42	 1.0666185e+00	 1.8029667e-01	 1.1140041e+00	 1.9237327e-01	[ 1.0077837e+00]	 2.1524119e-01
      43	 1.0852283e+00	 1.7567222e-01	 1.1328215e+00	 1.8817212e-01	[ 1.0242357e+00]	 1.8991852e-01


.. parsed-literal::

      44	 1.0992190e+00	 1.7087500e-01	 1.1462947e+00	 1.8466577e-01	[ 1.0470241e+00]	 2.1686888e-01
      45	 1.1200569e+00	 1.6880026e-01	 1.1682755e+00	 1.8279487e-01	[ 1.0633429e+00]	 1.7723823e-01


.. parsed-literal::

      46	 1.1329789e+00	 1.6830087e-01	 1.1812078e+00	 1.8267622e-01	[ 1.0749059e+00]	 1.7998838e-01


.. parsed-literal::

      47	 1.1516006e+00	 1.6638595e-01	 1.2006339e+00	 1.8201055e-01	[ 1.0899234e+00]	 2.0959139e-01


.. parsed-literal::

      48	 1.1592760e+00	 1.6527114e-01	 1.2087278e+00	 1.8075402e-01	[ 1.1090887e+00]	 2.0286250e-01


.. parsed-literal::

      49	 1.1724503e+00	 1.6374534e-01	 1.2212242e+00	 1.7973824e-01	[ 1.1173906e+00]	 2.1103239e-01


.. parsed-literal::

      50	 1.1774159e+00	 1.6246292e-01	 1.2262857e+00	 1.7858252e-01	[ 1.1201362e+00]	 2.1304393e-01


.. parsed-literal::

      51	 1.1925228e+00	 1.5924003e-01	 1.2422433e+00	 1.7573938e-01	[ 1.1265196e+00]	 2.1322942e-01


.. parsed-literal::

      52	 1.2057097e+00	 1.5569160e-01	 1.2557362e+00	 1.7191566e-01	[ 1.1351173e+00]	 2.0831800e-01
      53	 1.2192684e+00	 1.5367139e-01	 1.2695655e+00	 1.6933744e-01	[ 1.1446314e+00]	 1.8940639e-01


.. parsed-literal::

      54	 1.2324321e+00	 1.5146775e-01	 1.2828993e+00	 1.6591789e-01	[ 1.1595910e+00]	 2.1222854e-01


.. parsed-literal::

      55	 1.2432301e+00	 1.5051979e-01	 1.2934190e+00	 1.6416675e-01	[ 1.1720793e+00]	 2.2048068e-01


.. parsed-literal::

      56	 1.2533867e+00	 1.4941207e-01	 1.3034617e+00	 1.6314191e-01	[ 1.1792154e+00]	 2.1754789e-01


.. parsed-literal::

      57	 1.2678995e+00	 1.4560127e-01	 1.3185847e+00	 1.5810024e-01	[ 1.1896856e+00]	 2.2137117e-01


.. parsed-literal::

      58	 1.2770988e+00	 1.4298653e-01	 1.3273965e+00	 1.5533038e-01	[ 1.1989378e+00]	 2.1386290e-01
      59	 1.2871095e+00	 1.4172990e-01	 1.3373743e+00	 1.5356509e-01	[ 1.2119573e+00]	 1.8408203e-01


.. parsed-literal::

      60	 1.2956204e+00	 1.4050988e-01	 1.3462976e+00	 1.5207303e-01	[ 1.2188750e+00]	 2.1004629e-01


.. parsed-literal::

      61	 1.3047735e+00	 1.3931632e-01	 1.3556511e+00	 1.5091462e-01	[ 1.2253402e+00]	 2.0379186e-01


.. parsed-literal::

      62	 1.3165097e+00	 1.3686974e-01	 1.3679397e+00	 1.4743712e-01	[ 1.2314423e+00]	 2.1965694e-01


.. parsed-literal::

      63	 1.3255459e+00	 1.3675023e-01	 1.3767829e+00	 1.4711383e-01	[ 1.2382193e+00]	 2.1308160e-01


.. parsed-literal::

      64	 1.3308655e+00	 1.3662733e-01	 1.3818535e+00	 1.4700173e-01	[ 1.2451842e+00]	 2.0995498e-01
      65	 1.3395051e+00	 1.3605491e-01	 1.3907951e+00	 1.4538677e-01	[ 1.2526549e+00]	 1.9609427e-01


.. parsed-literal::

      66	 1.3484773e+00	 1.3502367e-01	 1.4000995e+00	 1.4376576e-01	[ 1.2605267e+00]	 2.0649147e-01


.. parsed-literal::

      67	 1.3585601e+00	 1.3444067e-01	 1.4104205e+00	 1.4186606e-01	[ 1.2717163e+00]	 2.1890450e-01


.. parsed-literal::

      68	 1.3686817e+00	 1.3381885e-01	 1.4204848e+00	 1.4112574e-01	[ 1.2780364e+00]	 2.0861530e-01


.. parsed-literal::

      69	 1.3748417e+00	 1.3335788e-01	 1.4266571e+00	 1.4107048e-01	  1.2762243e+00 	 2.1024990e-01
      70	 1.3811250e+00	 1.3282332e-01	 1.4329531e+00	 1.4074916e-01	  1.2747339e+00 	 1.7749572e-01


.. parsed-literal::

      71	 1.3878971e+00	 1.3207963e-01	 1.4396715e+00	 1.4028105e-01	  1.2742095e+00 	 2.1063685e-01
      72	 1.3935369e+00	 1.3177952e-01	 1.4451950e+00	 1.4016952e-01	  1.2772690e+00 	 1.8724155e-01


.. parsed-literal::

      73	 1.4013603e+00	 1.3158799e-01	 1.4529966e+00	 1.3998310e-01	[ 1.2861502e+00]	 1.8683839e-01


.. parsed-literal::

      74	 1.4060061e+00	 1.3121071e-01	 1.4575924e+00	 1.4014495e-01	  1.2827754e+00 	 2.1092582e-01
      75	 1.4116749e+00	 1.3110912e-01	 1.4631372e+00	 1.3985433e-01	[ 1.2963814e+00]	 2.0005989e-01


.. parsed-literal::

      76	 1.4168395e+00	 1.3093871e-01	 1.4684303e+00	 1.3952122e-01	[ 1.3040130e+00]	 2.0853996e-01


.. parsed-literal::

      77	 1.4210370e+00	 1.3069341e-01	 1.4728047e+00	 1.3930744e-01	[ 1.3054682e+00]	 2.2074056e-01


.. parsed-literal::

      78	 1.4295330e+00	 1.3023274e-01	 1.4817108e+00	 1.3904423e-01	  1.2975897e+00 	 2.1050787e-01


.. parsed-literal::

      79	 1.4334350e+00	 1.3001297e-01	 1.4857664e+00	 1.3877797e-01	  1.2966848e+00 	 3.1863475e-01


.. parsed-literal::

      80	 1.4380232e+00	 1.2957050e-01	 1.4904063e+00	 1.3847144e-01	  1.2913137e+00 	 2.0953107e-01
      81	 1.4421003e+00	 1.2923024e-01	 1.4945138e+00	 1.3819934e-01	  1.2919005e+00 	 1.7577720e-01


.. parsed-literal::

      82	 1.4463359e+00	 1.2881024e-01	 1.4988545e+00	 1.3764671e-01	  1.2970974e+00 	 2.1192002e-01


.. parsed-literal::

      83	 1.4506855e+00	 1.2845187e-01	 1.5033095e+00	 1.3723595e-01	  1.3034557e+00 	 2.0318675e-01


.. parsed-literal::

      84	 1.4539134e+00	 1.2827843e-01	 1.5065608e+00	 1.3690935e-01	[ 1.3079699e+00]	 2.0672917e-01
      85	 1.4577203e+00	 1.2799717e-01	 1.5104442e+00	 1.3652691e-01	  1.3065977e+00 	 1.8756700e-01


.. parsed-literal::

      86	 1.4613835e+00	 1.2805673e-01	 1.5141413e+00	 1.3655839e-01	  1.2988925e+00 	 1.9924808e-01
      87	 1.4649582e+00	 1.2802582e-01	 1.5176556e+00	 1.3653959e-01	  1.2954179e+00 	 1.9738984e-01


.. parsed-literal::

      88	 1.4691721e+00	 1.2790863e-01	 1.5218654e+00	 1.3698720e-01	  1.2833376e+00 	 1.7386723e-01


.. parsed-literal::

      89	 1.4721893e+00	 1.2763294e-01	 1.5248193e+00	 1.3693578e-01	  1.2856626e+00 	 2.0600653e-01


.. parsed-literal::

      90	 1.4753258e+00	 1.2728853e-01	 1.5279477e+00	 1.3702733e-01	  1.2882938e+00 	 2.1253872e-01


.. parsed-literal::

      91	 1.4799528e+00	 1.2681692e-01	 1.5327701e+00	 1.3739814e-01	  1.2780021e+00 	 2.0357752e-01
      92	 1.4815370e+00	 1.2676720e-01	 1.5345723e+00	 1.3817193e-01	  1.2762437e+00 	 1.9813013e-01


.. parsed-literal::

      93	 1.4854155e+00	 1.2669668e-01	 1.5383211e+00	 1.3786460e-01	  1.2774025e+00 	 2.1847749e-01
      94	 1.4875145e+00	 1.2678036e-01	 1.5404320e+00	 1.3790105e-01	  1.2706012e+00 	 1.8429184e-01


.. parsed-literal::

      95	 1.4901090e+00	 1.2684218e-01	 1.5431026e+00	 1.3792911e-01	  1.2662217e+00 	 2.1469235e-01


.. parsed-literal::

      96	 1.4930652e+00	 1.2692383e-01	 1.5462153e+00	 1.3807885e-01	  1.2550369e+00 	 2.0685720e-01


.. parsed-literal::

      97	 1.4964340e+00	 1.2687427e-01	 1.5495763e+00	 1.3807436e-01	  1.2573874e+00 	 2.0854354e-01


.. parsed-literal::

      98	 1.4987042e+00	 1.2674634e-01	 1.5518520e+00	 1.3794050e-01	  1.2641468e+00 	 2.1739554e-01
      99	 1.5009062e+00	 1.2662950e-01	 1.5540510e+00	 1.3774123e-01	  1.2660286e+00 	 1.9328022e-01


.. parsed-literal::

     100	 1.5036172e+00	 1.2657351e-01	 1.5568360e+00	 1.3757748e-01	  1.2690226e+00 	 2.1913123e-01
     101	 1.5062577e+00	 1.2655135e-01	 1.5594418e+00	 1.3728235e-01	  1.2712438e+00 	 1.8529105e-01


.. parsed-literal::

     102	 1.5092431e+00	 1.2641334e-01	 1.5624759e+00	 1.3699489e-01	  1.2698406e+00 	 2.1455669e-01
     103	 1.5117438e+00	 1.2619561e-01	 1.5650440e+00	 1.3685247e-01	  1.2696461e+00 	 1.9849014e-01


.. parsed-literal::

     104	 1.5140563e+00	 1.2592489e-01	 1.5673893e+00	 1.3672377e-01	  1.2708746e+00 	 2.0834398e-01
     105	 1.5172917e+00	 1.2547829e-01	 1.5707049e+00	 1.3673825e-01	  1.2689462e+00 	 1.8337560e-01


.. parsed-literal::

     106	 1.5196994e+00	 1.2533285e-01	 1.5732112e+00	 1.3658183e-01	  1.2664378e+00 	 1.9925809e-01
     107	 1.5218718e+00	 1.2530074e-01	 1.5753371e+00	 1.3650551e-01	  1.2670259e+00 	 1.7528820e-01


.. parsed-literal::

     108	 1.5251588e+00	 1.2525442e-01	 1.5786373e+00	 1.3623231e-01	  1.2665824e+00 	 2.1253085e-01
     109	 1.5271787e+00	 1.2520836e-01	 1.5807871e+00	 1.3600802e-01	  1.2656079e+00 	 1.8223572e-01


.. parsed-literal::

     110	 1.5291197e+00	 1.2511258e-01	 1.5827289e+00	 1.3585800e-01	  1.2665110e+00 	 2.0615816e-01


.. parsed-literal::

     111	 1.5318205e+00	 1.2482629e-01	 1.5855157e+00	 1.3559873e-01	  1.2609227e+00 	 2.0919561e-01
     112	 1.5339031e+00	 1.2460833e-01	 1.5876571e+00	 1.3545166e-01	  1.2532489e+00 	 1.9259810e-01


.. parsed-literal::

     113	 1.5358232e+00	 1.2425553e-01	 1.5897513e+00	 1.3515417e-01	  1.2384931e+00 	 2.0455050e-01


.. parsed-literal::

     114	 1.5376482e+00	 1.2417142e-01	 1.5915007e+00	 1.3518482e-01	  1.2302956e+00 	 2.0672870e-01


.. parsed-literal::

     115	 1.5385284e+00	 1.2418325e-01	 1.5923496e+00	 1.3519314e-01	  1.2298861e+00 	 2.0771646e-01


.. parsed-literal::

     116	 1.5404442e+00	 1.2417036e-01	 1.5943325e+00	 1.3519433e-01	  1.2234013e+00 	 2.0948839e-01
     117	 1.5428834e+00	 1.2421314e-01	 1.5968773e+00	 1.3525908e-01	  1.2154032e+00 	 1.8648052e-01


.. parsed-literal::

     118	 1.5446480e+00	 1.2429198e-01	 1.5987134e+00	 1.3531197e-01	  1.2081638e+00 	 2.0307589e-01


.. parsed-literal::

     119	 1.5464424e+00	 1.2425620e-01	 1.6004539e+00	 1.3525893e-01	  1.2078847e+00 	 2.0340633e-01
     120	 1.5484671e+00	 1.2421740e-01	 1.6025174e+00	 1.3518710e-01	  1.2063156e+00 	 1.8321896e-01


.. parsed-literal::

     121	 1.5499720e+00	 1.2418953e-01	 1.6040613e+00	 1.3515866e-01	  1.1980073e+00 	 2.1734595e-01


.. parsed-literal::

     122	 1.5518600e+00	 1.2416225e-01	 1.6059903e+00	 1.3508819e-01	  1.1961694e+00 	 2.1850753e-01


.. parsed-literal::

     123	 1.5535381e+00	 1.2418219e-01	 1.6077082e+00	 1.3507185e-01	  1.1953382e+00 	 2.1064138e-01


.. parsed-literal::

     124	 1.5549037e+00	 1.2418696e-01	 1.6091316e+00	 1.3504011e-01	  1.1902351e+00 	 2.0864582e-01
     125	 1.5568487e+00	 1.2425036e-01	 1.6111152e+00	 1.3510772e-01	  1.1861253e+00 	 1.9306254e-01


.. parsed-literal::

     126	 1.5585685e+00	 1.2418090e-01	 1.6128111e+00	 1.3508638e-01	  1.1780340e+00 	 2.0350218e-01


.. parsed-literal::

     127	 1.5596657e+00	 1.2413715e-01	 1.6138338e+00	 1.3517905e-01	  1.1770969e+00 	 2.1289945e-01


.. parsed-literal::

     128	 1.5607339e+00	 1.2407088e-01	 1.6148693e+00	 1.3526075e-01	  1.1764786e+00 	 2.0369935e-01
     129	 1.5620270e+00	 1.2403776e-01	 1.6161853e+00	 1.3547163e-01	  1.1729062e+00 	 1.8025804e-01


.. parsed-literal::

     130	 1.5636103e+00	 1.2402591e-01	 1.6178157e+00	 1.3561873e-01	  1.1701253e+00 	 2.1944785e-01
     131	 1.5655064e+00	 1.2409046e-01	 1.6198658e+00	 1.3576918e-01	  1.1577701e+00 	 1.9773984e-01


.. parsed-literal::

     132	 1.5671392e+00	 1.2428620e-01	 1.6216229e+00	 1.3595822e-01	  1.1522030e+00 	 2.1301889e-01


.. parsed-literal::

     133	 1.5685227e+00	 1.2423457e-01	 1.6229571e+00	 1.3582444e-01	  1.1486072e+00 	 2.0197058e-01


.. parsed-literal::

     134	 1.5699988e+00	 1.2422108e-01	 1.6244296e+00	 1.3577201e-01	  1.1409839e+00 	 2.1041441e-01


.. parsed-literal::

     135	 1.5712289e+00	 1.2416973e-01	 1.6256630e+00	 1.3582815e-01	  1.1374530e+00 	 2.0409036e-01
     136	 1.5725979e+00	 1.2396444e-01	 1.6271720e+00	 1.3593544e-01	  1.1375047e+00 	 2.0122623e-01


.. parsed-literal::

     137	 1.5744512e+00	 1.2391720e-01	 1.6289711e+00	 1.3613226e-01	  1.1296629e+00 	 2.1358895e-01


.. parsed-literal::

     138	 1.5754809e+00	 1.2384611e-01	 1.6299873e+00	 1.3619220e-01	  1.1295037e+00 	 2.1083713e-01
     139	 1.5769824e+00	 1.2372974e-01	 1.6315670e+00	 1.3633992e-01	  1.1200980e+00 	 1.7832065e-01


.. parsed-literal::

     140	 1.5779245e+00	 1.2368135e-01	 1.6325894e+00	 1.3650199e-01	  1.1147749e+00 	 3.2336092e-01
     141	 1.5791744e+00	 1.2362843e-01	 1.6338994e+00	 1.3651581e-01	  1.1044636e+00 	 1.7860746e-01


.. parsed-literal::

     142	 1.5804672e+00	 1.2365208e-01	 1.6352618e+00	 1.3659956e-01	  1.0907799e+00 	 2.1070480e-01


.. parsed-literal::

     143	 1.5817772e+00	 1.2372244e-01	 1.6365998e+00	 1.3663403e-01	  1.0802897e+00 	 2.0357084e-01
     144	 1.5832089e+00	 1.2382196e-01	 1.6380763e+00	 1.3681820e-01	  1.0656925e+00 	 1.9444346e-01


.. parsed-literal::

     145	 1.5843150e+00	 1.2389076e-01	 1.6391180e+00	 1.3691780e-01	  1.0603279e+00 	 2.0035934e-01
     146	 1.5851760e+00	 1.2384821e-01	 1.6399011e+00	 1.3687324e-01	  1.0550756e+00 	 1.7629957e-01


.. parsed-literal::

     147	 1.5864022e+00	 1.2375428e-01	 1.6410931e+00	 1.3691123e-01	  1.0426319e+00 	 2.0272446e-01
     148	 1.5874826e+00	 1.2361214e-01	 1.6422126e+00	 1.3700770e-01	  1.0018237e+00 	 1.6950345e-01


.. parsed-literal::

     149	 1.5884699e+00	 1.2356044e-01	 1.6432010e+00	 1.3700777e-01	  1.0032655e+00 	 2.0471597e-01


.. parsed-literal::

     150	 1.5894422e+00	 1.2351772e-01	 1.6442237e+00	 1.3707803e-01	  9.9619135e-01 	 2.1488261e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.02 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc990c44eb0>



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
    CPU times: user 1.78 s, sys: 33 ms, total: 1.81 s
    Wall time: 576 ms


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

