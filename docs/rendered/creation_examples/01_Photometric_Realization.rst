Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fea904227d0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.162990  0.139958  
    1      25.391064  0.005864  0.004885  
    2      24.304707  0.142035  0.130830  
    3      25.291103  0.128344  0.069274  
    4      25.096743  0.029782  0.017601  
    ...          ...       ...       ...  
    99995  24.737946  0.207536  0.126818  
    99996  24.224169  0.197746  0.110998  
    99997  25.613836  0.000927  0.000878  
    99998  25.274899  0.156492  0.098850  
    99999  25.699642  0.100060  0.062815  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  input: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.055151</td>
          <td>0.570900</td>
          <td>27.386419</td>
          <td>0.290908</td>
          <td>25.982839</td>
          <td>0.077661</td>
          <td>25.163039</td>
          <td>0.061352</td>
          <td>24.638820</td>
          <td>0.073817</td>
          <td>24.180414</td>
          <td>0.110679</td>
          <td>0.162990</td>
          <td>0.139958</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.872111</td>
          <td>0.499809</td>
          <td>27.207096</td>
          <td>0.251394</td>
          <td>26.617098</td>
          <td>0.135248</td>
          <td>26.145030</td>
          <td>0.145092</td>
          <td>26.057433</td>
          <td>0.249895</td>
          <td>25.617104</td>
          <td>0.366980</td>
          <td>0.005864</td>
          <td>0.004885</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.599493</td>
          <td>0.307523</td>
          <td>26.080244</td>
          <td>0.137215</td>
          <td>25.020678</td>
          <td>0.103298</td>
          <td>24.315079</td>
          <td>0.124437</td>
          <td>0.142035</td>
          <td>0.130830</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.686658</td>
          <td>0.874240</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.374382</td>
          <td>0.256222</td>
          <td>25.987892</td>
          <td>0.126681</td>
          <td>25.637409</td>
          <td>0.175900</td>
          <td>25.461156</td>
          <td>0.324524</td>
          <td>0.128344</td>
          <td>0.069274</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.793649</td>
          <td>0.471551</td>
          <td>26.239968</td>
          <td>0.110541</td>
          <td>25.975926</td>
          <td>0.077188</td>
          <td>25.734446</td>
          <td>0.101576</td>
          <td>25.578901</td>
          <td>0.167363</td>
          <td>24.983862</td>
          <td>0.219942</td>
          <td>0.029782</td>
          <td>0.017601</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>31.240366</td>
          <td>3.816985</td>
          <td>26.358729</td>
          <td>0.122567</td>
          <td>25.400775</td>
          <td>0.046353</td>
          <td>25.101764</td>
          <td>0.058106</td>
          <td>24.809642</td>
          <td>0.085827</td>
          <td>24.780111</td>
          <td>0.185378</td>
          <td>0.207536</td>
          <td>0.126818</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.985177</td>
          <td>1.049252</td>
          <td>26.501266</td>
          <td>0.138643</td>
          <td>26.067941</td>
          <td>0.083717</td>
          <td>25.199560</td>
          <td>0.063372</td>
          <td>24.834263</td>
          <td>0.087708</td>
          <td>24.165223</td>
          <td>0.109222</td>
          <td>0.197746</td>
          <td>0.110998</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.977907</td>
          <td>0.540004</td>
          <td>26.562457</td>
          <td>0.146137</td>
          <td>26.344305</td>
          <td>0.106703</td>
          <td>26.582067</td>
          <td>0.210261</td>
          <td>25.752592</td>
          <td>0.193893</td>
          <td>25.514841</td>
          <td>0.338640</td>
          <td>0.000927</td>
          <td>0.000878</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.548841</td>
          <td>0.391561</td>
          <td>26.012079</td>
          <td>0.090558</td>
          <td>26.131435</td>
          <td>0.088531</td>
          <td>25.838535</td>
          <td>0.111250</td>
          <td>25.790683</td>
          <td>0.200205</td>
          <td>25.516512</td>
          <td>0.339088</td>
          <td>0.156492</td>
          <td>0.098850</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.074808</td>
          <td>0.268767</td>
          <td>26.751959</td>
          <td>0.171825</td>
          <td>26.678275</td>
          <td>0.142575</td>
          <td>26.231397</td>
          <td>0.156254</td>
          <td>26.265672</td>
          <td>0.296045</td>
          <td>25.636619</td>
          <td>0.372610</td>
          <td>0.100060</td>
          <td>0.062815</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.929418</td>
          <td>1.139043</td>
          <td>26.872107</td>
          <td>0.232034</td>
          <td>26.115819</td>
          <td>0.110334</td>
          <td>25.230106</td>
          <td>0.083219</td>
          <td>24.957130</td>
          <td>0.123289</td>
          <td>23.966424</td>
          <td>0.116731</td>
          <td>0.162990</td>
          <td>0.139958</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.752103</td>
          <td>0.988654</td>
          <td>27.200536</td>
          <td>0.285508</td>
          <td>26.466126</td>
          <td>0.139158</td>
          <td>26.535401</td>
          <td>0.237204</td>
          <td>25.698887</td>
          <td>0.216215</td>
          <td>25.388988</td>
          <td>0.356181</td>
          <td>0.005864</td>
          <td>0.004885</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.297833</td>
          <td>0.681726</td>
          <td>28.866484</td>
          <td>0.916124</td>
          <td>25.883072</td>
          <td>0.145080</td>
          <td>25.138183</td>
          <td>0.142331</td>
          <td>24.541849</td>
          <td>0.188755</td>
          <td>0.142035</td>
          <td>0.130830</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.260364</td>
          <td>0.652653</td>
          <td>27.211614</td>
          <td>0.269172</td>
          <td>26.080341</td>
          <td>0.167487</td>
          <td>25.349827</td>
          <td>0.166501</td>
          <td>24.814975</td>
          <td>0.231457</td>
          <td>0.128344</td>
          <td>0.069274</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.873154</td>
          <td>0.254624</td>
          <td>26.039959</td>
          <td>0.107219</td>
          <td>25.888439</td>
          <td>0.084227</td>
          <td>25.811665</td>
          <td>0.128660</td>
          <td>25.571988</td>
          <td>0.194766</td>
          <td>24.616402</td>
          <td>0.189914</td>
          <td>0.029782</td>
          <td>0.017601</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.279085</td>
          <td>0.374802</td>
          <td>26.618079</td>
          <td>0.190531</td>
          <td>25.574916</td>
          <td>0.069774</td>
          <td>24.932387</td>
          <td>0.065125</td>
          <td>24.972364</td>
          <td>0.127072</td>
          <td>24.729485</td>
          <td>0.227471</td>
          <td>0.207536</td>
          <td>0.126818</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.883633</td>
          <td>0.583963</td>
          <td>26.492758</td>
          <td>0.169693</td>
          <td>25.887714</td>
          <td>0.090920</td>
          <td>25.128957</td>
          <td>0.076604</td>
          <td>24.696563</td>
          <td>0.098837</td>
          <td>24.454504</td>
          <td>0.178670</td>
          <td>0.197746</td>
          <td>0.110998</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.097399</td>
          <td>0.304903</td>
          <td>26.733129</td>
          <td>0.194055</td>
          <td>26.345045</td>
          <td>0.125315</td>
          <td>26.179659</td>
          <td>0.176045</td>
          <td>25.586378</td>
          <td>0.196755</td>
          <td>24.825972</td>
          <td>0.225880</td>
          <td>0.000927</td>
          <td>0.000878</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.321703</td>
          <td>0.377951</td>
          <td>26.143181</td>
          <td>0.122965</td>
          <td>25.968804</td>
          <td>0.095279</td>
          <td>26.322106</td>
          <td>0.209523</td>
          <td>25.967650</td>
          <td>0.283772</td>
          <td>25.742972</td>
          <td>0.489908</td>
          <td>0.156492</td>
          <td>0.098850</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.932617</td>
          <td>0.233795</td>
          <td>26.477363</td>
          <td>0.143719</td>
          <td>26.500763</td>
          <td>0.235730</td>
          <td>26.009040</td>
          <td>0.285068</td>
          <td>26.102369</td>
          <td>0.618402</td>
          <td>0.100060</td>
          <td>0.062815</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.509113</td>
          <td>0.172038</td>
          <td>26.110671</td>
          <td>0.110504</td>
          <td>25.114518</td>
          <td>0.075625</td>
          <td>24.726251</td>
          <td>0.101427</td>
          <td>23.860093</td>
          <td>0.107056</td>
          <td>0.162990</td>
          <td>0.139958</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.168068</td>
          <td>0.618592</td>
          <td>28.390424</td>
          <td>0.622630</td>
          <td>26.414514</td>
          <td>0.113490</td>
          <td>26.377478</td>
          <td>0.177042</td>
          <td>25.471098</td>
          <td>0.152691</td>
          <td>25.222422</td>
          <td>0.267839</td>
          <td>0.005864</td>
          <td>0.004885</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.497513</td>
          <td>0.428787</td>
          <td>27.986302</td>
          <td>0.540202</td>
          <td>27.959039</td>
          <td>0.485522</td>
          <td>26.022311</td>
          <td>0.160529</td>
          <td>25.224565</td>
          <td>0.150619</td>
          <td>24.499635</td>
          <td>0.178925</td>
          <td>0.142035</td>
          <td>0.130830</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.021440</td>
          <td>1.136157</td>
          <td>27.516739</td>
          <td>0.355913</td>
          <td>27.256479</td>
          <td>0.260739</td>
          <td>26.243885</td>
          <td>0.178734</td>
          <td>25.529726</td>
          <td>0.180658</td>
          <td>25.494756</td>
          <td>0.373205</td>
          <td>0.128344</td>
          <td>0.069274</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.753737</td>
          <td>0.207308</td>
          <td>26.219464</td>
          <td>0.109340</td>
          <td>25.928300</td>
          <td>0.074610</td>
          <td>25.515476</td>
          <td>0.084512</td>
          <td>25.759403</td>
          <td>0.196515</td>
          <td>24.902727</td>
          <td>0.207176</td>
          <td>0.029782</td>
          <td>0.017601</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.471810</td>
          <td>0.172944</td>
          <td>25.438762</td>
          <td>0.063731</td>
          <td>25.004713</td>
          <td>0.071578</td>
          <td>24.954491</td>
          <td>0.128828</td>
          <td>24.790347</td>
          <td>0.246066</td>
          <td>0.207536</td>
          <td>0.126818</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.263776</td>
          <td>1.379555</td>
          <td>26.803544</td>
          <td>0.222492</td>
          <td>26.146010</td>
          <td>0.115113</td>
          <td>25.303730</td>
          <td>0.090256</td>
          <td>24.689396</td>
          <td>0.099206</td>
          <td>24.036118</td>
          <td>0.126030</td>
          <td>0.197746</td>
          <td>0.110998</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.043649</td>
          <td>0.219620</td>
          <td>26.435747</td>
          <td>0.115565</td>
          <td>26.416937</td>
          <td>0.182990</td>
          <td>25.550684</td>
          <td>0.163386</td>
          <td>25.855372</td>
          <td>0.440786</td>
          <td>0.000927</td>
          <td>0.000878</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.049223</td>
          <td>1.189794</td>
          <td>26.033543</td>
          <td>0.108732</td>
          <td>26.164787</td>
          <td>0.109641</td>
          <td>25.755452</td>
          <td>0.125224</td>
          <td>25.967719</td>
          <td>0.275746</td>
          <td>25.675188</td>
          <td>0.453202</td>
          <td>0.156492</td>
          <td>0.098850</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.340765</td>
          <td>0.351444</td>
          <td>26.704741</td>
          <td>0.177466</td>
          <td>26.540216</td>
          <td>0.137631</td>
          <td>26.874392</td>
          <td>0.290667</td>
          <td>25.908045</td>
          <td>0.239330</td>
          <td>25.717752</td>
          <td>0.428532</td>
          <td>0.100060</td>
          <td>0.062815</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
