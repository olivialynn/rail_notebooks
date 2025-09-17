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
       1	-3.4098004e-01	 3.1960296e-01	-3.3127982e-01	 3.2426256e-01	[-3.3942839e-01]	 4.6190214e-01


.. parsed-literal::

       2	-2.6857333e-01	 3.0843499e-01	-2.4411153e-01	 3.1361062e-01	[-2.5700581e-01]	 2.3228598e-01


.. parsed-literal::

       3	-2.2230247e-01	 2.8697041e-01	-1.7878201e-01	 2.9240660e-01	[-1.9914626e-01]	 2.8460288e-01


.. parsed-literal::

       4	-1.8151954e-01	 2.6907260e-01	-1.3246088e-01	 2.7632850e-01	[-1.6777012e-01]	 2.8852081e-01


.. parsed-literal::

       5	-1.2228557e-01	 2.5589657e-01	-9.2435841e-02	 2.6503409e-01	[-1.3468535e-01]	 2.1192575e-01


.. parsed-literal::

       6	-6.5916370e-02	 2.5144795e-01	-3.8679336e-02	 2.5824948e-01	[-6.4225602e-02]	 2.1064901e-01


.. parsed-literal::

       7	-4.7454417e-02	 2.4757540e-01	-2.4056457e-02	 2.5517155e-01	[-4.9952887e-02]	 2.0815492e-01


.. parsed-literal::

       8	-3.5659179e-02	 2.4568171e-01	-1.5515047e-02	 2.5288892e-01	[-4.2345817e-02]	 2.1059871e-01
       9	-2.1120303e-02	 2.4296511e-01	-3.7288147e-03	 2.5063234e-01	[-3.1539307e-02]	 1.7042613e-01


.. parsed-literal::

      10	-1.1549070e-02	 2.4115321e-01	 4.0443859e-03	 2.4802752e-01	[-2.4757640e-02]	 2.1279860e-01


.. parsed-literal::

      11	-7.0710520e-03	 2.4035146e-01	 7.7894188e-03	 2.4732000e-01	[-2.1262345e-02]	 2.0765185e-01


.. parsed-literal::

      12	-2.6050710e-03	 2.3957121e-01	 1.1656395e-02	 2.4642065e-01	[-1.6299244e-02]	 2.1502161e-01
      13	 2.8310888e-03	 2.3847479e-01	 1.7137505e-02	 2.4555100e-01	[-1.0595202e-02]	 1.7627668e-01


.. parsed-literal::

      14	 8.9180629e-02	 2.2441131e-01	 1.0889844e-01	 2.3662078e-01	[ 8.3362301e-02]	 3.2902098e-01


.. parsed-literal::

      15	 1.2097662e-01	 2.2273012e-01	 1.4285907e-01	 2.2942866e-01	[ 1.2406158e-01]	 2.0999908e-01


.. parsed-literal::

      16	 1.9400609e-01	 2.1653197e-01	 2.1838273e-01	 2.2518975e-01	[ 1.9028438e-01]	 2.0486641e-01


.. parsed-literal::

      17	 3.0582020e-01	 2.1131016e-01	 3.3783716e-01	 2.1872810e-01	[ 2.9244742e-01]	 2.0633531e-01
      18	 3.5138484e-01	 2.0744834e-01	 3.8500509e-01	 2.1540669e-01	[ 3.3244176e-01]	 2.0256376e-01


.. parsed-literal::

      19	 4.0431272e-01	 2.0491065e-01	 4.3847311e-01	 2.1237864e-01	[ 3.9584791e-01]	 1.9930267e-01
      20	 4.5833853e-01	 2.0073241e-01	 4.9254599e-01	 2.0988844e-01	[ 4.4681574e-01]	 1.8145990e-01


.. parsed-literal::

      21	 5.6436712e-01	 1.9519691e-01	 5.9940050e-01	 2.0549403e-01	[ 5.5122048e-01]	 2.0239139e-01
      22	 6.1889283e-01	 1.9881398e-01	 6.5858394e-01	 2.0841840e-01	[ 6.2457350e-01]	 1.7477155e-01


.. parsed-literal::

      23	 6.7487557e-01	 1.9001228e-01	 7.1341464e-01	 2.0158072e-01	[ 6.8035801e-01]	 1.9741392e-01
      24	 6.9622620e-01	 1.8643949e-01	 7.3374959e-01	 1.9638259e-01	[ 7.0166888e-01]	 1.7182136e-01


.. parsed-literal::

      25	 7.2055607e-01	 1.8545871e-01	 7.5765507e-01	 1.9362451e-01	[ 7.2607904e-01]	 2.0725799e-01
      26	 7.5035252e-01	 1.8909718e-01	 7.8697273e-01	 1.9425607e-01	[ 7.6191725e-01]	 1.9862461e-01


.. parsed-literal::

      27	 7.9445695e-01	 1.8503733e-01	 8.3232947e-01	 1.8881914e-01	[ 8.0671310e-01]	 2.0963001e-01
      28	 8.2438890e-01	 1.8489797e-01	 8.6349855e-01	 1.8722827e-01	[ 8.4529848e-01]	 1.7865181e-01


.. parsed-literal::

      29	 8.4984857e-01	 1.8440682e-01	 8.9033720e-01	 1.8460198e-01	[ 8.6768121e-01]	 2.0876813e-01


.. parsed-literal::

      30	 8.7198504e-01	 1.8463005e-01	 9.1246263e-01	 1.8592243e-01	[ 8.7880059e-01]	 2.1170187e-01


.. parsed-literal::

      31	 8.9141891e-01	 1.8264019e-01	 9.3211293e-01	 1.8496196e-01	[ 8.9301609e-01]	 2.0368004e-01
      32	 9.1710097e-01	 1.8272029e-01	 9.5841981e-01	 1.8655225e-01	[ 9.0421181e-01]	 1.9853854e-01


.. parsed-literal::

      33	 9.4051943e-01	 1.7994511e-01	 9.8267161e-01	 1.8200328e-01	[ 9.2495537e-01]	 2.0933247e-01


.. parsed-literal::

      34	 9.6311951e-01	 1.7636408e-01	 1.0051074e+00	 1.7962289e-01	[ 9.4176531e-01]	 2.0615506e-01
      35	 9.7961528e-01	 1.7463431e-01	 1.0219517e+00	 1.7782343e-01	[ 9.6207450e-01]	 1.9633079e-01


.. parsed-literal::

      36	 9.9945606e-01	 1.7240761e-01	 1.0427175e+00	 1.7563122e-01	[ 9.7977520e-01]	 1.9059110e-01


.. parsed-literal::

      37	 1.0270611e+00	 1.6705981e-01	 1.0726036e+00	 1.7062116e-01	[ 9.9196642e-01]	 2.1978235e-01


.. parsed-literal::

      38	 1.0400977e+00	 1.6021852e-01	 1.0868461e+00	 1.6807446e-01	  9.7897465e-01 	 2.1120787e-01


.. parsed-literal::

      39	 1.0556596e+00	 1.5991895e-01	 1.1017631e+00	 1.6756183e-01	[ 9.9780613e-01]	 2.1383810e-01


.. parsed-literal::

      40	 1.0670966e+00	 1.5845435e-01	 1.1132832e+00	 1.6703239e-01	[ 1.0054866e+00]	 2.0844436e-01
      41	 1.0811206e+00	 1.5640989e-01	 1.1273418e+00	 1.6641220e-01	[ 1.0154794e+00]	 1.9972134e-01


.. parsed-literal::

      42	 1.0950590e+00	 1.5225601e-01	 1.1416844e+00	 1.6529607e-01	[ 1.0228379e+00]	 2.0730209e-01
      43	 1.1078341e+00	 1.5017921e-01	 1.1542828e+00	 1.6469074e-01	[ 1.0309090e+00]	 1.8208003e-01


.. parsed-literal::

      44	 1.1179647e+00	 1.4911188e-01	 1.1645196e+00	 1.6287207e-01	[ 1.0398997e+00]	 2.0366240e-01


.. parsed-literal::

      45	 1.1322068e+00	 1.4612254e-01	 1.1797554e+00	 1.5955693e-01	  1.0335395e+00 	 2.0178032e-01


.. parsed-literal::

      46	 1.1408260e+00	 1.4625404e-01	 1.1889388e+00	 1.5748192e-01	  1.0364714e+00 	 2.1050048e-01
      47	 1.1525958e+00	 1.4436031e-01	 1.2004669e+00	 1.5664192e-01	[ 1.0442326e+00]	 1.7847705e-01


.. parsed-literal::

      48	 1.1636262e+00	 1.4238419e-01	 1.2116166e+00	 1.5585584e-01	[ 1.0504537e+00]	 2.0564914e-01


.. parsed-literal::

      49	 1.1740840e+00	 1.4105184e-01	 1.2222367e+00	 1.5479027e-01	[ 1.0626430e+00]	 2.0734358e-01


.. parsed-literal::

      50	 1.1907569e+00	 1.3905738e-01	 1.2395016e+00	 1.5325808e-01	[ 1.0819860e+00]	 2.1710038e-01


.. parsed-literal::

      51	 1.1974112e+00	 1.3910457e-01	 1.2465824e+00	 1.5206168e-01	[ 1.0886558e+00]	 2.0147109e-01
      52	 1.2089895e+00	 1.3825100e-01	 1.2577798e+00	 1.5137555e-01	[ 1.1020805e+00]	 1.9846630e-01


.. parsed-literal::

      53	 1.2167980e+00	 1.3787615e-01	 1.2657616e+00	 1.5132164e-01	[ 1.1040519e+00]	 1.7715931e-01


.. parsed-literal::

      54	 1.2267693e+00	 1.3776393e-01	 1.2761854e+00	 1.5085766e-01	[ 1.1064737e+00]	 2.0613027e-01
      55	 1.2369053e+00	 1.3784428e-01	 1.2867056e+00	 1.5039327e-01	[ 1.1119336e+00]	 1.9483924e-01


.. parsed-literal::

      56	 1.2472664e+00	 1.3778954e-01	 1.2973749e+00	 1.4944056e-01	[ 1.1201340e+00]	 2.0667577e-01


.. parsed-literal::

      57	 1.2566198e+00	 1.3684982e-01	 1.3067977e+00	 1.4795866e-01	[ 1.1272323e+00]	 2.1979427e-01


.. parsed-literal::

      58	 1.2668157e+00	 1.3574532e-01	 1.3172196e+00	 1.4639434e-01	[ 1.1314994e+00]	 2.1838474e-01


.. parsed-literal::

      59	 1.2746396e+00	 1.3495142e-01	 1.3253388e+00	 1.4518483e-01	  1.1143808e+00 	 2.0513153e-01


.. parsed-literal::

      60	 1.2822797e+00	 1.3428028e-01	 1.3326980e+00	 1.4509409e-01	  1.1163878e+00 	 2.0503044e-01


.. parsed-literal::

      61	 1.2910232e+00	 1.3357268e-01	 1.3415319e+00	 1.4522995e-01	  1.1027005e+00 	 2.0458603e-01
      62	 1.2991258e+00	 1.3316275e-01	 1.3498196e+00	 1.4539649e-01	  1.0821886e+00 	 1.7415452e-01


.. parsed-literal::

      63	 1.3085155e+00	 1.3258558e-01	 1.3595031e+00	 1.4573880e-01	  9.9293073e-01 	 2.1035004e-01


.. parsed-literal::

      64	 1.3173575e+00	 1.3240555e-01	 1.3683662e+00	 1.4526340e-01	  9.8794484e-01 	 2.0683312e-01


.. parsed-literal::

      65	 1.3238411e+00	 1.3189159e-01	 1.3750524e+00	 1.4356426e-01	  9.8549006e-01 	 2.1481633e-01


.. parsed-literal::

      66	 1.3311676e+00	 1.3135703e-01	 1.3826176e+00	 1.4133253e-01	  9.7032658e-01 	 2.0986843e-01
      67	 1.3370293e+00	 1.3125788e-01	 1.3886059e+00	 1.3931391e-01	  9.2506870e-01 	 2.0435882e-01


.. parsed-literal::

      68	 1.3427834e+00	 1.3091862e-01	 1.3942213e+00	 1.3940189e-01	  9.1841475e-01 	 2.1463680e-01


.. parsed-literal::

      69	 1.3480449e+00	 1.3077423e-01	 1.3995615e+00	 1.3963146e-01	  8.9964587e-01 	 2.1242857e-01


.. parsed-literal::

      70	 1.3531593e+00	 1.3045545e-01	 1.4048198e+00	 1.3951444e-01	  8.6421347e-01 	 2.0437694e-01


.. parsed-literal::

      71	 1.3611260e+00	 1.3026607e-01	 1.4129779e+00	 1.3947309e-01	  8.3401875e-01 	 2.0702052e-01


.. parsed-literal::

      72	 1.3681714e+00	 1.2988590e-01	 1.4203308e+00	 1.3889594e-01	  8.0671937e-01 	 2.0526528e-01
      73	 1.3728291e+00	 1.2989180e-01	 1.4251924e+00	 1.3893804e-01	  7.5083492e-01 	 1.9614530e-01


.. parsed-literal::

      74	 1.3768830e+00	 1.2932719e-01	 1.4289474e+00	 1.3848645e-01	  7.7534763e-01 	 2.0977879e-01
      75	 1.3810775e+00	 1.2890007e-01	 1.4332774e+00	 1.3777828e-01	  7.5980823e-01 	 1.7919660e-01


.. parsed-literal::

      76	 1.3868530e+00	 1.2834487e-01	 1.4394199e+00	 1.3695619e-01	  7.0448700e-01 	 2.0839214e-01


.. parsed-literal::

      77	 1.3907064e+00	 1.2802376e-01	 1.4436488e+00	 1.3579447e-01	  6.4798144e-01 	 2.1140122e-01


.. parsed-literal::

      78	 1.3961470e+00	 1.2804446e-01	 1.4489157e+00	 1.3575203e-01	  6.6153888e-01 	 2.1138525e-01


.. parsed-literal::

      79	 1.3994118e+00	 1.2800100e-01	 1.4521306e+00	 1.3559760e-01	  6.6206993e-01 	 2.0140386e-01


.. parsed-literal::

      80	 1.4036472e+00	 1.2796025e-01	 1.4565241e+00	 1.3569651e-01	  6.6225680e-01 	 2.1194744e-01


.. parsed-literal::

      81	 1.4084725e+00	 1.2739464e-01	 1.4614788e+00	 1.3568635e-01	  6.7578769e-01 	 2.0974588e-01


.. parsed-literal::

      82	 1.4129669e+00	 1.2749410e-01	 1.4660514e+00	 1.3623578e-01	  6.9418646e-01 	 2.1719551e-01


.. parsed-literal::

      83	 1.4157639e+00	 1.2739515e-01	 1.4687812e+00	 1.3624437e-01	  6.9222763e-01 	 2.0554352e-01


.. parsed-literal::

      84	 1.4209854e+00	 1.2702148e-01	 1.4741354e+00	 1.3604821e-01	  6.6616652e-01 	 2.1012735e-01
      85	 1.4231044e+00	 1.2684580e-01	 1.4765003e+00	 1.3558767e-01	  6.5154348e-01 	 2.0010686e-01


.. parsed-literal::

      86	 1.4264896e+00	 1.2665987e-01	 1.4797526e+00	 1.3536588e-01	  6.4955115e-01 	 2.1789503e-01
      87	 1.4289879e+00	 1.2657241e-01	 1.4822417e+00	 1.3499321e-01	  6.5140702e-01 	 1.7752051e-01


.. parsed-literal::

      88	 1.4317868e+00	 1.2641535e-01	 1.4850837e+00	 1.3449522e-01	  6.4038988e-01 	 2.1283174e-01


.. parsed-literal::

      89	 1.4362210e+00	 1.2637827e-01	 1.4896632e+00	 1.3353689e-01	  6.3742425e-01 	 2.0989728e-01


.. parsed-literal::

      90	 1.4393974e+00	 1.2609969e-01	 1.4928794e+00	 1.3286798e-01	  5.9274055e-01 	 2.1189976e-01
      91	 1.4423997e+00	 1.2589364e-01	 1.4957860e+00	 1.3283663e-01	  5.9126295e-01 	 1.9271350e-01


.. parsed-literal::

      92	 1.4455477e+00	 1.2571492e-01	 1.4989733e+00	 1.3283439e-01	  5.9095262e-01 	 1.9108295e-01


.. parsed-literal::

      93	 1.4482582e+00	 1.2530104e-01	 1.5016593e+00	 1.3271921e-01	  5.9522368e-01 	 2.1418738e-01


.. parsed-literal::

      94	 1.4517256e+00	 1.2514395e-01	 1.5051397e+00	 1.3296634e-01	  6.1386831e-01 	 2.1439648e-01
      95	 1.4541509e+00	 1.2492178e-01	 1.5075147e+00	 1.3298332e-01	  6.4768282e-01 	 1.8673539e-01


.. parsed-literal::

      96	 1.4561160e+00	 1.2482875e-01	 1.5094160e+00	 1.3298541e-01	  6.5536568e-01 	 2.0751476e-01
      97	 1.4581656e+00	 1.2473493e-01	 1.5114737e+00	 1.3282800e-01	  6.6343851e-01 	 1.8126321e-01


.. parsed-literal::

      98	 1.4612622e+00	 1.2454205e-01	 1.5146757e+00	 1.3251182e-01	  6.7289725e-01 	 1.9117332e-01
      99	 1.4630139e+00	 1.2458840e-01	 1.5166633e+00	 1.3261595e-01	  7.0453545e-01 	 1.8894506e-01


.. parsed-literal::

     100	 1.4666111e+00	 1.2437793e-01	 1.5201920e+00	 1.3241966e-01	  7.0063021e-01 	 1.9823170e-01


.. parsed-literal::

     101	 1.4684887e+00	 1.2428311e-01	 1.5220813e+00	 1.3247644e-01	  7.0036743e-01 	 2.2578645e-01
     102	 1.4708049e+00	 1.2418394e-01	 1.5244643e+00	 1.3269009e-01	  7.0096159e-01 	 1.8935227e-01


.. parsed-literal::

     103	 1.4748218e+00	 1.2394247e-01	 1.5286612e+00	 1.3314460e-01	  6.9746785e-01 	 2.2878146e-01


.. parsed-literal::

     104	 1.4769384e+00	 1.2394185e-01	 1.5309204e+00	 1.3328276e-01	  7.0822872e-01 	 3.1489635e-01


.. parsed-literal::

     105	 1.4792903e+00	 1.2378534e-01	 1.5333302e+00	 1.3331894e-01	  7.1351858e-01 	 2.1476316e-01


.. parsed-literal::

     106	 1.4813005e+00	 1.2369603e-01	 1.5354386e+00	 1.3306514e-01	  7.2184268e-01 	 2.0951653e-01
     107	 1.4834635e+00	 1.2355726e-01	 1.5377234e+00	 1.3243066e-01	  7.4984922e-01 	 1.9645786e-01


.. parsed-literal::

     108	 1.4860490e+00	 1.2359900e-01	 1.5403880e+00	 1.3209499e-01	  7.7859241e-01 	 1.9822955e-01


.. parsed-literal::

     109	 1.4886332e+00	 1.2368676e-01	 1.5429997e+00	 1.3182198e-01	  8.1666471e-01 	 2.0424271e-01


.. parsed-literal::

     110	 1.4903319e+00	 1.2375041e-01	 1.5446693e+00	 1.3183982e-01	  8.4140910e-01 	 2.0209289e-01
     111	 1.4921383e+00	 1.2370124e-01	 1.5464041e+00	 1.3198534e-01	  8.5218517e-01 	 1.8759823e-01


.. parsed-literal::

     112	 1.4946325e+00	 1.2362113e-01	 1.5488853e+00	 1.3242109e-01	  8.6359764e-01 	 2.1121049e-01


.. parsed-literal::

     113	 1.4967111e+00	 1.2363387e-01	 1.5510273e+00	 1.3269910e-01	  8.7143608e-01 	 2.1305060e-01


.. parsed-literal::

     114	 1.4987930e+00	 1.2361799e-01	 1.5531660e+00	 1.3289888e-01	  8.7291392e-01 	 2.1579218e-01


.. parsed-literal::

     115	 1.5010497e+00	 1.2355225e-01	 1.5554969e+00	 1.3286820e-01	  8.6730739e-01 	 2.2155428e-01


.. parsed-literal::

     116	 1.5033720e+00	 1.2343331e-01	 1.5579739e+00	 1.3269732e-01	  8.6380214e-01 	 2.1206713e-01


.. parsed-literal::

     117	 1.5057257e+00	 1.2328123e-01	 1.5604066e+00	 1.3231717e-01	  8.5315816e-01 	 2.1891069e-01


.. parsed-literal::

     118	 1.5078210e+00	 1.2313027e-01	 1.5624813e+00	 1.3207300e-01	  8.5911684e-01 	 2.0693636e-01


.. parsed-literal::

     119	 1.5098505e+00	 1.2311707e-01	 1.5644929e+00	 1.3197405e-01	  8.6120895e-01 	 2.0362973e-01


.. parsed-literal::

     120	 1.5113222e+00	 1.2301526e-01	 1.5660055e+00	 1.3193323e-01	  8.5892390e-01 	 2.0686936e-01


.. parsed-literal::

     121	 1.5128380e+00	 1.2298148e-01	 1.5675539e+00	 1.3199364e-01	  8.4911227e-01 	 2.1710253e-01


.. parsed-literal::

     122	 1.5152168e+00	 1.2286626e-01	 1.5701229e+00	 1.3210684e-01	  8.2333944e-01 	 2.0265532e-01
     123	 1.5163706e+00	 1.2279956e-01	 1.5713838e+00	 1.3213469e-01	  8.1424965e-01 	 2.0611668e-01


.. parsed-literal::

     124	 1.5177548e+00	 1.2269030e-01	 1.5727502e+00	 1.3206092e-01	  8.1441014e-01 	 2.0584488e-01
     125	 1.5202310e+00	 1.2229405e-01	 1.5752520e+00	 1.3188489e-01	  8.1475991e-01 	 1.9565821e-01


.. parsed-literal::

     126	 1.5216316e+00	 1.2213436e-01	 1.5766295e+00	 1.3189072e-01	  8.1672201e-01 	 1.7281246e-01


.. parsed-literal::

     127	 1.5234261e+00	 1.2193836e-01	 1.5783755e+00	 1.3191050e-01	  8.1994577e-01 	 2.0675492e-01
     128	 1.5256766e+00	 1.2153368e-01	 1.5807155e+00	 1.3233569e-01	  8.1407286e-01 	 1.7432928e-01


.. parsed-literal::

     129	 1.5270269e+00	 1.2153945e-01	 1.5820491e+00	 1.3218261e-01	  8.1876566e-01 	 1.7849302e-01


.. parsed-literal::

     130	 1.5280245e+00	 1.2158970e-01	 1.5830086e+00	 1.3213318e-01	  8.1816733e-01 	 2.1332312e-01


.. parsed-literal::

     131	 1.5297847e+00	 1.2159655e-01	 1.5848238e+00	 1.3219183e-01	  8.0883228e-01 	 2.1482372e-01
     132	 1.5314593e+00	 1.2157081e-01	 1.5865699e+00	 1.3225155e-01	  8.0143157e-01 	 1.9065046e-01


.. parsed-literal::

     133	 1.5327228e+00	 1.2142105e-01	 1.5879324e+00	 1.3233259e-01	  7.8810773e-01 	 2.0186925e-01
     134	 1.5349168e+00	 1.2141163e-01	 1.5900423e+00	 1.3230541e-01	  7.9297978e-01 	 1.8595862e-01


.. parsed-literal::

     135	 1.5362688e+00	 1.2132288e-01	 1.5913266e+00	 1.3219691e-01	  8.0139997e-01 	 2.1564150e-01
     136	 1.5377071e+00	 1.2126248e-01	 1.5927305e+00	 1.3205947e-01	  8.0212598e-01 	 1.8670464e-01


.. parsed-literal::

     137	 1.5389929e+00	 1.2096830e-01	 1.5940844e+00	 1.3158309e-01	  8.1726019e-01 	 2.0824265e-01


.. parsed-literal::

     138	 1.5404062e+00	 1.2097735e-01	 1.5954897e+00	 1.3155774e-01	  8.0694586e-01 	 2.0648551e-01


.. parsed-literal::

     139	 1.5416823e+00	 1.2094687e-01	 1.5968603e+00	 1.3147261e-01	  7.9303012e-01 	 2.0238924e-01
     140	 1.5428070e+00	 1.2087855e-01	 1.5980761e+00	 1.3137693e-01	  7.8325135e-01 	 1.8702769e-01


.. parsed-literal::

     141	 1.5439737e+00	 1.2078579e-01	 1.5993673e+00	 1.3125571e-01	  7.7434556e-01 	 2.9226041e-01


.. parsed-literal::

     142	 1.5456122e+00	 1.2064465e-01	 1.6010937e+00	 1.3111987e-01	  7.6796845e-01 	 2.0520687e-01


.. parsed-literal::

     143	 1.5466000e+00	 1.2056578e-01	 1.6020572e+00	 1.3108087e-01	  7.7342695e-01 	 2.0730829e-01
     144	 1.5479961e+00	 1.2045205e-01	 1.6034042e+00	 1.3105959e-01	  7.8561944e-01 	 2.0186329e-01


.. parsed-literal::

     145	 1.5491059e+00	 1.2038125e-01	 1.6044429e+00	 1.3098574e-01	  8.0014620e-01 	 2.0512676e-01


.. parsed-literal::

     146	 1.5500100e+00	 1.2034700e-01	 1.6053178e+00	 1.3097913e-01	  8.0227970e-01 	 2.0590639e-01


.. parsed-literal::

     147	 1.5513541e+00	 1.2027128e-01	 1.6066852e+00	 1.3095264e-01	  8.0254836e-01 	 2.0563626e-01


.. parsed-literal::

     148	 1.5522753e+00	 1.2017654e-01	 1.6076332e+00	 1.3100955e-01	  8.0112211e-01 	 2.1179152e-01
     149	 1.5536487e+00	 1.2012877e-01	 1.6091768e+00	 1.3114595e-01	  7.7739784e-01 	 2.0303869e-01


.. parsed-literal::

     150	 1.5549269e+00	 1.1996693e-01	 1.6104613e+00	 1.3123274e-01	  7.8214679e-01 	 2.2841811e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.1 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe00cef1d80>



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
    CPU times: user 2.08 s, sys: 47.9 ms, total: 2.13 s
    Wall time: 637 ms


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

