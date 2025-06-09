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
       1	-3.4896362e-01	 3.2237032e-01	-3.3922118e-01	 3.1321304e-01	[-3.2149004e-01]	 4.6111012e-01


.. parsed-literal::

       2	-2.7925743e-01	 3.1204597e-01	-2.5542543e-01	 3.0307127e-01	[-2.2852651e-01]	 2.2834277e-01


.. parsed-literal::

       3	-2.3637542e-01	 2.9182719e-01	-1.9467996e-01	 2.8386615e-01	[-1.6299379e-01]	 2.7794671e-01
       4	-1.9856093e-01	 2.6859366e-01	-1.5582437e-01	 2.6236794e-01	[-1.1794093e-01]	 2.0046091e-01


.. parsed-literal::

       5	-1.1378920e-01	 2.5928614e-01	-7.9926789e-02	 2.5023024e-01	[-3.8803075e-02]	 2.0248580e-01


.. parsed-literal::

       6	-7.9567480e-02	 2.5410418e-01	-4.9748499e-02	 2.4498904e-01	[-1.7604606e-02]	 2.0579815e-01


.. parsed-literal::

       7	-6.2302790e-02	 2.5126407e-01	-3.7943074e-02	 2.4170617e-01	[-1.8083833e-03]	 2.1274638e-01
       8	-4.8587108e-02	 2.4893699e-01	-2.8298330e-02	 2.3919218e-01	[ 9.7741287e-03]	 1.8432832e-01


.. parsed-literal::

       9	-3.5505449e-02	 2.4649765e-01	-1.7959449e-02	 2.3699457e-01	[ 1.9336831e-02]	 1.7933726e-01
      10	-2.3007342e-02	 2.4389277e-01	-7.5399848e-03	 2.3468181e-01	[ 3.1198962e-02]	 1.8423486e-01


.. parsed-literal::

      11	-2.2904130e-02	 2.4320808e-01	-9.0238428e-03	 2.3462630e-01	  2.0834229e-02 	 2.1515346e-01
      12	-1.4806024e-02	 2.4273505e-01	-9.6508709e-04	 2.3399690e-01	[ 3.3788404e-02]	 1.7265439e-01


.. parsed-literal::

      13	-1.2501933e-02	 2.4223913e-01	 1.4066983e-03	 2.3342540e-01	[ 3.7388037e-02]	 1.7221141e-01
      14	-8.1876328e-03	 2.4119880e-01	 6.3659328e-03	 2.3250240e-01	[ 4.2859758e-02]	 1.8050957e-01


.. parsed-literal::

      15	 2.1228453e-01	 2.2403633e-01	 2.3754855e-01	 2.1958321e-01	[ 2.5254407e-01]	 4.0817618e-01


.. parsed-literal::

      16	 3.0966041e-01	 2.1491210e-01	 3.3771324e-01	 2.0993197e-01	[ 3.5609823e-01]	 2.1073532e-01
      17	 4.1703408e-01	 2.1075669e-01	 4.4991093e-01	 2.0552497e-01	[ 4.6838790e-01]	 1.7587280e-01


.. parsed-literal::

      18	 4.7607736e-01	 2.0850639e-01	 5.1059968e-01	 2.0523444e-01	[ 5.1424862e-01]	 1.8354511e-01


.. parsed-literal::

      19	 5.3589243e-01	 2.0348745e-01	 5.7123559e-01	 2.0093599e-01	[ 5.6821062e-01]	 2.1634674e-01


.. parsed-literal::

      20	 5.9099151e-01	 2.0053577e-01	 6.2946534e-01	 2.0007583e-01	[ 6.0624737e-01]	 2.1684170e-01


.. parsed-literal::

      21	 6.3941474e-01	 1.9587871e-01	 6.7841091e-01	 1.9549475e-01	[ 6.4875922e-01]	 2.0614529e-01
      22	 6.6894687e-01	 1.9256912e-01	 7.0640394e-01	 1.9218009e-01	[ 6.8587120e-01]	 1.7794633e-01


.. parsed-literal::

      23	 6.9010095e-01	 1.9154395e-01	 7.2778931e-01	 1.9218584e-01	[ 7.0026025e-01]	 1.8869710e-01


.. parsed-literal::

      24	 7.2392601e-01	 1.9284937e-01	 7.6251954e-01	 1.9393673e-01	[ 7.2206072e-01]	 2.1294165e-01
      25	 7.5081986e-01	 1.9234080e-01	 7.8933395e-01	 1.9321422e-01	[ 7.5545250e-01]	 2.0766521e-01


.. parsed-literal::

      26	 7.7384256e-01	 1.8725718e-01	 8.1248502e-01	 1.8862128e-01	[ 7.7566395e-01]	 2.0754361e-01


.. parsed-literal::

      27	 8.0497314e-01	 1.8481859e-01	 8.4372941e-01	 1.8662947e-01	[ 8.0282014e-01]	 2.0510817e-01


.. parsed-literal::

      28	 8.3154780e-01	 1.8497168e-01	 8.7157526e-01	 1.8386733e-01	[ 8.3144757e-01]	 2.0266914e-01


.. parsed-literal::

      29	 8.5312576e-01	 1.8408247e-01	 8.9342712e-01	 1.8277647e-01	[ 8.5637700e-01]	 2.0080757e-01


.. parsed-literal::

      30	 8.7190343e-01	 1.8327575e-01	 9.1228440e-01	 1.8198507e-01	[ 8.7945057e-01]	 2.0592332e-01
      31	 8.9109592e-01	 1.8159263e-01	 9.3167232e-01	 1.8041431e-01	[ 9.0576324e-01]	 1.9680715e-01


.. parsed-literal::

      32	 9.1465668e-01	 1.8118909e-01	 9.5603286e-01	 1.8205974e-01	[ 9.3296788e-01]	 2.0198965e-01
      33	 9.3074818e-01	 1.7873429e-01	 9.7255949e-01	 1.7962511e-01	[ 9.5322897e-01]	 1.9943500e-01


.. parsed-literal::

      34	 9.4268047e-01	 1.7675989e-01	 9.8473469e-01	 1.7868410e-01	[ 9.6323362e-01]	 2.0940042e-01
      35	 9.5217239e-01	 1.7559049e-01	 9.9626173e-01	 1.7996489e-01	  9.5017475e-01 	 1.9810939e-01


.. parsed-literal::

      36	 9.7127437e-01	 1.7325615e-01	 1.0149118e+00	 1.7867463e-01	[ 9.7451781e-01]	 2.0556021e-01


.. parsed-literal::

      37	 9.7762237e-01	 1.7266982e-01	 1.0213023e+00	 1.7851678e-01	[ 9.8175457e-01]	 2.0789862e-01


.. parsed-literal::

      38	 9.9983832e-01	 1.6984495e-01	 1.0443367e+00	 1.7552449e-01	[ 1.0050447e+00]	 2.0966911e-01
      39	 1.0127930e+00	 1.6975726e-01	 1.0582150e+00	 1.7466731e-01	[ 1.0227057e+00]	 1.8607569e-01


.. parsed-literal::

      40	 1.0285438e+00	 1.6800421e-01	 1.0740308e+00	 1.7284634e-01	[ 1.0313427e+00]	 2.0106745e-01


.. parsed-literal::

      41	 1.0378921e+00	 1.6646379e-01	 1.0838178e+00	 1.7175991e-01	[ 1.0344553e+00]	 2.0683122e-01
      42	 1.0502642e+00	 1.6485485e-01	 1.0965175e+00	 1.7053954e-01	[ 1.0426476e+00]	 1.9877243e-01


.. parsed-literal::

      43	 1.0582581e+00	 1.6456271e-01	 1.1050522e+00	 1.6974820e-01	[ 1.0505285e+00]	 1.7868996e-01


.. parsed-literal::

      44	 1.0728384e+00	 1.6262595e-01	 1.1191121e+00	 1.6878755e-01	[ 1.0740299e+00]	 2.0963764e-01
      45	 1.0809135e+00	 1.6093261e-01	 1.1271875e+00	 1.6748251e-01	[ 1.0834446e+00]	 1.7634988e-01


.. parsed-literal::

      46	 1.0935087e+00	 1.5759055e-01	 1.1403075e+00	 1.6484213e-01	[ 1.0982305e+00]	 2.0796943e-01
      47	 1.1037134e+00	 1.5528809e-01	 1.1506981e+00	 1.6232242e-01	[ 1.1001317e+00]	 2.0117140e-01


.. parsed-literal::

      48	 1.1143145e+00	 1.5366308e-01	 1.1613957e+00	 1.6095433e-01	[ 1.1113510e+00]	 2.0601988e-01


.. parsed-literal::

      49	 1.1285436e+00	 1.5143119e-01	 1.1758245e+00	 1.5915279e-01	[ 1.1293850e+00]	 2.0557857e-01


.. parsed-literal::

      50	 1.1392015e+00	 1.4982195e-01	 1.1864438e+00	 1.5742306e-01	[ 1.1393785e+00]	 2.0585895e-01
      51	 1.1530280e+00	 1.4737973e-01	 1.2005168e+00	 1.5492288e-01	[ 1.1532651e+00]	 1.9923377e-01


.. parsed-literal::

      52	 1.1620803e+00	 1.4673563e-01	 1.2097679e+00	 1.5411429e-01	[ 1.1699135e+00]	 2.1657515e-01


.. parsed-literal::

      53	 1.1706247e+00	 1.4582293e-01	 1.2184187e+00	 1.5337831e-01	[ 1.1818510e+00]	 2.2754240e-01


.. parsed-literal::

      54	 1.1835026e+00	 1.4412126e-01	 1.2317646e+00	 1.5197115e-01	[ 1.1987595e+00]	 2.0917988e-01


.. parsed-literal::

      55	 1.1918629e+00	 1.4246285e-01	 1.2405890e+00	 1.5084001e-01	[ 1.2065127e+00]	 2.1127510e-01


.. parsed-literal::

      56	 1.1996757e+00	 1.4170326e-01	 1.2485286e+00	 1.4971487e-01	[ 1.2174138e+00]	 2.0283055e-01


.. parsed-literal::

      57	 1.2047526e+00	 1.4171801e-01	 1.2534215e+00	 1.4936344e-01	[ 1.2183765e+00]	 2.0339918e-01


.. parsed-literal::

      58	 1.2119078e+00	 1.4157559e-01	 1.2607550e+00	 1.4873213e-01	[ 1.2231582e+00]	 2.0651150e-01
      59	 1.2248595e+00	 1.4157221e-01	 1.2743252e+00	 1.4735005e-01	[ 1.2293450e+00]	 1.9946146e-01


.. parsed-literal::

      60	 1.2309948e+00	 1.4165361e-01	 1.2805243e+00	 1.4634063e-01	[ 1.2359549e+00]	 3.3315659e-01
      61	 1.2385510e+00	 1.4101192e-01	 1.2882088e+00	 1.4552145e-01	[ 1.2427477e+00]	 2.0062613e-01


.. parsed-literal::

      62	 1.2444673e+00	 1.4025258e-01	 1.2942981e+00	 1.4475397e-01	[ 1.2486713e+00]	 2.2231889e-01


.. parsed-literal::

      63	 1.2504788e+00	 1.3930957e-01	 1.3005478e+00	 1.4394156e-01	[ 1.2490780e+00]	 2.1216869e-01


.. parsed-literal::

      64	 1.2572133e+00	 1.3897980e-01	 1.3074977e+00	 1.4348112e-01	[ 1.2516726e+00]	 2.1607995e-01


.. parsed-literal::

      65	 1.2648196e+00	 1.3829692e-01	 1.3151286e+00	 1.4278110e-01	[ 1.2559283e+00]	 2.1429539e-01


.. parsed-literal::

      66	 1.2734002e+00	 1.3792023e-01	 1.3241163e+00	 1.4144880e-01	  1.2554223e+00 	 2.0735407e-01


.. parsed-literal::

      67	 1.2800232e+00	 1.3796038e-01	 1.3307332e+00	 1.4123849e-01	[ 1.2674753e+00]	 2.1041870e-01


.. parsed-literal::

      68	 1.2849482e+00	 1.3770760e-01	 1.3356037e+00	 1.4072394e-01	[ 1.2718443e+00]	 2.0419502e-01


.. parsed-literal::

      69	 1.2941941e+00	 1.3742392e-01	 1.3448801e+00	 1.3999992e-01	[ 1.2849945e+00]	 2.0571256e-01
      70	 1.2998732e+00	 1.3674993e-01	 1.3507958e+00	 1.3890853e-01	[ 1.2884025e+00]	 1.9632792e-01


.. parsed-literal::

      71	 1.3058291e+00	 1.3647703e-01	 1.3566386e+00	 1.3872650e-01	[ 1.2964170e+00]	 2.0718122e-01


.. parsed-literal::

      72	 1.3121671e+00	 1.3572419e-01	 1.3630670e+00	 1.3811911e-01	[ 1.3040356e+00]	 2.1092725e-01


.. parsed-literal::

      73	 1.3172065e+00	 1.3498209e-01	 1.3681708e+00	 1.3763187e-01	[ 1.3087577e+00]	 2.0950603e-01


.. parsed-literal::

      74	 1.3237143e+00	 1.3379266e-01	 1.3750099e+00	 1.3674428e-01	[ 1.3273086e+00]	 2.0550871e-01
      75	 1.3312825e+00	 1.3328535e-01	 1.3825969e+00	 1.3597548e-01	[ 1.3320921e+00]	 1.9525266e-01


.. parsed-literal::

      76	 1.3358929e+00	 1.3315814e-01	 1.3872648e+00	 1.3570431e-01	[ 1.3371326e+00]	 2.1584296e-01
      77	 1.3431906e+00	 1.3279447e-01	 1.3948762e+00	 1.3504980e-01	[ 1.3434259e+00]	 1.9741201e-01


.. parsed-literal::

      78	 1.3487916e+00	 1.3214930e-01	 1.4008322e+00	 1.3437667e-01	[ 1.3503179e+00]	 2.0176768e-01


.. parsed-literal::

      79	 1.3561754e+00	 1.3162473e-01	 1.4081568e+00	 1.3365978e-01	[ 1.3570286e+00]	 2.1669006e-01


.. parsed-literal::

      80	 1.3617911e+00	 1.3101182e-01	 1.4137325e+00	 1.3302235e-01	[ 1.3615641e+00]	 2.1917057e-01


.. parsed-literal::

      81	 1.3682168e+00	 1.3038164e-01	 1.4202372e+00	 1.3276884e-01	[ 1.3675344e+00]	 2.1815681e-01


.. parsed-literal::

      82	 1.3741661e+00	 1.2961150e-01	 1.4263250e+00	 1.3218914e-01	[ 1.3754823e+00]	 2.8620815e-01


.. parsed-literal::

      83	 1.3800333e+00	 1.2923010e-01	 1.4320748e+00	 1.3218717e-01	[ 1.3781687e+00]	 2.1827602e-01


.. parsed-literal::

      84	 1.3840942e+00	 1.2884520e-01	 1.4361532e+00	 1.3198098e-01	[ 1.3829816e+00]	 2.0688915e-01
      85	 1.3889592e+00	 1.2822616e-01	 1.4410805e+00	 1.3145104e-01	[ 1.3834778e+00]	 1.7792702e-01


.. parsed-literal::

      86	 1.3928041e+00	 1.2707395e-01	 1.4451091e+00	 1.3043059e-01	[ 1.3879309e+00]	 2.1804070e-01
      87	 1.3981047e+00	 1.2689135e-01	 1.4503575e+00	 1.2998349e-01	[ 1.3886513e+00]	 1.8023753e-01


.. parsed-literal::

      88	 1.4015204e+00	 1.2655898e-01	 1.4537991e+00	 1.2944399e-01	[ 1.3901632e+00]	 2.0516586e-01
      89	 1.4051703e+00	 1.2610716e-01	 1.4575523e+00	 1.2880828e-01	[ 1.3932375e+00]	 1.8954349e-01


.. parsed-literal::

      90	 1.4103563e+00	 1.2529005e-01	 1.4629120e+00	 1.2788358e-01	  1.3922037e+00 	 2.1345329e-01


.. parsed-literal::

      91	 1.4143518e+00	 1.2470017e-01	 1.4671681e+00	 1.2712206e-01	[ 1.4005598e+00]	 2.1275425e-01
      92	 1.4186651e+00	 1.2463909e-01	 1.4712716e+00	 1.2745540e-01	[ 1.4014235e+00]	 1.9820380e-01


.. parsed-literal::

      93	 1.4217149e+00	 1.2439989e-01	 1.4743196e+00	 1.2748515e-01	  1.4011019e+00 	 2.1396518e-01


.. parsed-literal::

      94	 1.4259503e+00	 1.2415219e-01	 1.4785982e+00	 1.2773000e-01	[ 1.4040169e+00]	 2.1847725e-01


.. parsed-literal::

      95	 1.4277715e+00	 1.2396579e-01	 1.4806198e+00	 1.2807707e-01	  1.4005711e+00 	 2.1028638e-01


.. parsed-literal::

      96	 1.4311762e+00	 1.2387437e-01	 1.4838278e+00	 1.2793043e-01	[ 1.4069142e+00]	 2.1653414e-01


.. parsed-literal::

      97	 1.4329133e+00	 1.2372536e-01	 1.4855482e+00	 1.2784601e-01	[ 1.4092510e+00]	 2.0860243e-01


.. parsed-literal::

      98	 1.4360875e+00	 1.2341057e-01	 1.4887624e+00	 1.2782516e-01	[ 1.4114083e+00]	 2.0722699e-01


.. parsed-literal::

      99	 1.4371553e+00	 1.2316780e-01	 1.4900947e+00	 1.2836416e-01	  1.4092438e+00 	 2.1079755e-01


.. parsed-literal::

     100	 1.4414255e+00	 1.2298912e-01	 1.4942067e+00	 1.2814234e-01	[ 1.4127284e+00]	 2.0215726e-01
     101	 1.4434520e+00	 1.2285151e-01	 1.4962241e+00	 1.2809243e-01	[ 1.4138356e+00]	 1.9159865e-01


.. parsed-literal::

     102	 1.4459080e+00	 1.2264839e-01	 1.4986900e+00	 1.2805491e-01	[ 1.4158620e+00]	 2.0821285e-01


.. parsed-literal::

     103	 1.4494376e+00	 1.2245065e-01	 1.5021904e+00	 1.2808657e-01	[ 1.4197554e+00]	 2.0355487e-01


.. parsed-literal::

     104	 1.4521182e+00	 1.2218774e-01	 1.5049351e+00	 1.2816788e-01	[ 1.4234621e+00]	 3.2617140e-01


.. parsed-literal::

     105	 1.4558158e+00	 1.2202020e-01	 1.5086128e+00	 1.2821504e-01	[ 1.4275720e+00]	 2.1218324e-01


.. parsed-literal::

     106	 1.4579779e+00	 1.2192514e-01	 1.5107713e+00	 1.2823563e-01	  1.4272012e+00 	 2.1404409e-01


.. parsed-literal::

     107	 1.4610482e+00	 1.2161898e-01	 1.5139676e+00	 1.2809662e-01	[ 1.4298033e+00]	 2.1144915e-01


.. parsed-literal::

     108	 1.4633252e+00	 1.2144609e-01	 1.5163278e+00	 1.2821738e-01	  1.4241243e+00 	 2.1122527e-01


.. parsed-literal::

     109	 1.4654857e+00	 1.2136603e-01	 1.5184920e+00	 1.2836105e-01	  1.4243707e+00 	 2.1287370e-01


.. parsed-literal::

     110	 1.4679574e+00	 1.2122562e-01	 1.5209707e+00	 1.2860438e-01	  1.4256904e+00 	 2.1499801e-01
     111	 1.4700335e+00	 1.2122120e-01	 1.5230062e+00	 1.2895519e-01	  1.4255116e+00 	 1.9817305e-01


.. parsed-literal::

     112	 1.4726833e+00	 1.2125196e-01	 1.5255896e+00	 1.2933814e-01	  1.4292571e+00 	 1.7896366e-01
     113	 1.4748190e+00	 1.2128162e-01	 1.5277391e+00	 1.2951877e-01	[ 1.4302852e+00]	 1.8643570e-01


.. parsed-literal::

     114	 1.4772380e+00	 1.2127270e-01	 1.5302732e+00	 1.2944406e-01	[ 1.4310478e+00]	 2.1119142e-01


.. parsed-literal::

     115	 1.4790496e+00	 1.2111494e-01	 1.5321667e+00	 1.2887580e-01	[ 1.4347873e+00]	 2.1827269e-01
     116	 1.4806112e+00	 1.2100300e-01	 1.5337845e+00	 1.2870343e-01	[ 1.4365441e+00]	 1.8348932e-01


.. parsed-literal::

     117	 1.4825462e+00	 1.2076487e-01	 1.5357424e+00	 1.2815730e-01	[ 1.4396161e+00]	 2.0031047e-01


.. parsed-literal::

     118	 1.4846904e+00	 1.2039121e-01	 1.5378823e+00	 1.2751220e-01	  1.4394764e+00 	 2.1161270e-01


.. parsed-literal::

     119	 1.4874564e+00	 1.2011677e-01	 1.5406566e+00	 1.2700092e-01	[ 1.4438403e+00]	 2.0860147e-01
     120	 1.4897237e+00	 1.1990107e-01	 1.5429082e+00	 1.2685486e-01	  1.4401398e+00 	 2.0346332e-01


.. parsed-literal::

     121	 1.4918863e+00	 1.1981034e-01	 1.5451325e+00	 1.2691265e-01	  1.4405935e+00 	 2.0148349e-01
     122	 1.4934383e+00	 1.1987642e-01	 1.5467979e+00	 1.2744950e-01	  1.4375954e+00 	 1.8416333e-01


.. parsed-literal::

     123	 1.4948484e+00	 1.1985006e-01	 1.5482396e+00	 1.2751028e-01	  1.4373826e+00 	 2.0639277e-01


.. parsed-literal::

     124	 1.4963626e+00	 1.1972864e-01	 1.5498707e+00	 1.2737087e-01	  1.4363098e+00 	 2.0763659e-01


.. parsed-literal::

     125	 1.4978217e+00	 1.1961520e-01	 1.5513955e+00	 1.2730279e-01	  1.4352276e+00 	 2.1391940e-01
     126	 1.4995854e+00	 1.1948420e-01	 1.5531590e+00	 1.2727528e-01	  1.4310919e+00 	 1.8222594e-01


.. parsed-literal::

     127	 1.5011660e+00	 1.1939353e-01	 1.5546816e+00	 1.2721845e-01	  1.4302208e+00 	 2.0775008e-01


.. parsed-literal::

     128	 1.5025347e+00	 1.1932954e-01	 1.5560527e+00	 1.2732310e-01	  1.4250833e+00 	 2.1441031e-01
     129	 1.5042209e+00	 1.1926176e-01	 1.5576984e+00	 1.2713507e-01	  1.4282658e+00 	 1.9695997e-01


.. parsed-literal::

     130	 1.5054017e+00	 1.1923044e-01	 1.5589029e+00	 1.2705133e-01	  1.4300650e+00 	 2.2126603e-01


.. parsed-literal::

     131	 1.5076470e+00	 1.1921128e-01	 1.5612857e+00	 1.2685162e-01	  1.4303141e+00 	 2.1799064e-01


.. parsed-literal::

     132	 1.5085034e+00	 1.1912748e-01	 1.5623073e+00	 1.2657825e-01	  1.4224934e+00 	 2.0905995e-01
     133	 1.5106207e+00	 1.1916234e-01	 1.5643630e+00	 1.2661545e-01	  1.4258805e+00 	 1.8038154e-01


.. parsed-literal::

     134	 1.5117155e+00	 1.1916096e-01	 1.5654375e+00	 1.2655577e-01	  1.4251644e+00 	 1.9367385e-01


.. parsed-literal::

     135	 1.5129247e+00	 1.1913897e-01	 1.5666256e+00	 1.2644688e-01	  1.4243241e+00 	 2.0636415e-01
     136	 1.5136593e+00	 1.1904961e-01	 1.5673338e+00	 1.2619714e-01	  1.4211058e+00 	 2.0021534e-01


.. parsed-literal::

     137	 1.5150992e+00	 1.1897790e-01	 1.5687499e+00	 1.2615906e-01	  1.4220856e+00 	 1.9661832e-01


.. parsed-literal::

     138	 1.5158635e+00	 1.1888452e-01	 1.5695034e+00	 1.2614253e-01	  1.4235438e+00 	 2.0841622e-01
     139	 1.5169551e+00	 1.1874319e-01	 1.5706100e+00	 1.2610411e-01	  1.4245764e+00 	 1.7645979e-01


.. parsed-literal::

     140	 1.5184740e+00	 1.1862575e-01	 1.5721550e+00	 1.2610133e-01	  1.4246379e+00 	 2.1425009e-01
     141	 1.5198714e+00	 1.1856365e-01	 1.5736594e+00	 1.2618402e-01	  1.4211733e+00 	 1.8939924e-01


.. parsed-literal::

     142	 1.5209873e+00	 1.1857170e-01	 1.5747562e+00	 1.2616573e-01	  1.4220173e+00 	 2.1715069e-01


.. parsed-literal::

     143	 1.5219305e+00	 1.1860570e-01	 1.5757043e+00	 1.2621596e-01	  1.4208440e+00 	 2.2000027e-01
     144	 1.5231076e+00	 1.1855690e-01	 1.5769191e+00	 1.2623231e-01	  1.4177092e+00 	 1.8649769e-01


.. parsed-literal::

     145	 1.5242298e+00	 1.1853430e-01	 1.5782156e+00	 1.2656652e-01	  1.4159197e+00 	 2.1388173e-01
     146	 1.5258044e+00	 1.1834322e-01	 1.5797452e+00	 1.2639900e-01	  1.4131821e+00 	 1.9977808e-01


.. parsed-literal::

     147	 1.5266996e+00	 1.1819641e-01	 1.5806291e+00	 1.2633415e-01	  1.4133972e+00 	 1.9158792e-01
     148	 1.5278707e+00	 1.1800918e-01	 1.5818210e+00	 1.2632237e-01	  1.4136540e+00 	 1.8494129e-01


.. parsed-literal::

     149	 1.5281901e+00	 1.1778305e-01	 1.5822927e+00	 1.2637384e-01	  1.4151743e+00 	 2.0207334e-01
     150	 1.5301836e+00	 1.1771217e-01	 1.5842141e+00	 1.2638409e-01	  1.4156143e+00 	 1.9900680e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.13 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4b68a80d00>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.81 s, sys: 42 ms, total: 1.85 s
    Wall time: 616 ms


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

