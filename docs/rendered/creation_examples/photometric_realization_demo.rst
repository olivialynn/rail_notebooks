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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fb937cc6ce0>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>28.793924</td>
          <td>1.615794</td>
          <td>26.678707</td>
          <td>0.161433</td>
          <td>26.104738</td>
          <td>0.086475</td>
          <td>25.298527</td>
          <td>0.069180</td>
          <td>25.068984</td>
          <td>0.107753</td>
          <td>24.561720</td>
          <td>0.153930</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.785517</td>
          <td>0.468697</td>
          <td>28.239603</td>
          <td>0.559238</td>
          <td>27.822638</td>
          <td>0.366928</td>
          <td>26.919282</td>
          <td>0.277660</td>
          <td>26.809658</td>
          <td>0.452594</td>
          <td>25.497686</td>
          <td>0.334073</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.585749</td>
          <td>0.402851</td>
          <td>25.951703</td>
          <td>0.085878</td>
          <td>24.798969</td>
          <td>0.027253</td>
          <td>23.854548</td>
          <td>0.019467</td>
          <td>23.167011</td>
          <td>0.020241</td>
          <td>22.867267</td>
          <td>0.034681</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.931568</td>
          <td>0.892012</td>
          <td>27.669884</td>
          <td>0.325301</td>
          <td>26.630660</td>
          <td>0.218964</td>
          <td>25.844594</td>
          <td>0.209459</td>
          <td>26.221437</td>
          <td>0.577054</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.794508</td>
          <td>0.213345</td>
          <td>25.676599</td>
          <td>0.067375</td>
          <td>25.373549</td>
          <td>0.045246</td>
          <td>24.823534</td>
          <td>0.045387</td>
          <td>24.308780</td>
          <td>0.055097</td>
          <td>23.699927</td>
          <td>0.072547</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.937463</td>
          <td>0.524351</td>
          <td>26.336318</td>
          <td>0.120206</td>
          <td>26.147328</td>
          <td>0.089778</td>
          <td>26.114227</td>
          <td>0.141295</td>
          <td>26.062392</td>
          <td>0.250915</td>
          <td>24.997730</td>
          <td>0.222495</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.877642</td>
          <td>0.191108</td>
          <td>26.667735</td>
          <td>0.141287</td>
          <td>26.351911</td>
          <td>0.173171</td>
          <td>26.477494</td>
          <td>0.350434</td>
          <td>25.473891</td>
          <td>0.327826</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.933063</td>
          <td>1.017274</td>
          <td>27.398398</td>
          <td>0.293732</td>
          <td>26.910631</td>
          <td>0.173928</td>
          <td>26.515599</td>
          <td>0.198865</td>
          <td>26.197761</td>
          <td>0.280232</td>
          <td>25.006048</td>
          <td>0.224039</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.597933</td>
          <td>0.344391</td>
          <td>26.880336</td>
          <td>0.169505</td>
          <td>25.717943</td>
          <td>0.100118</td>
          <td>25.409114</td>
          <td>0.144720</td>
          <td>25.853130</td>
          <td>0.440035</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.328512</td>
          <td>0.690999</td>
          <td>26.659566</td>
          <td>0.158816</td>
          <td>26.158616</td>
          <td>0.090673</td>
          <td>25.724797</td>
          <td>0.100721</td>
          <td>25.193944</td>
          <td>0.120148</td>
          <td>24.668988</td>
          <td>0.168700</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>26.963549</td>
          <td>0.588204</td>
          <td>27.112453</td>
          <td>0.265780</td>
          <td>25.969910</td>
          <td>0.090305</td>
          <td>25.286363</td>
          <td>0.081096</td>
          <td>25.021289</td>
          <td>0.121317</td>
          <td>24.507349</td>
          <td>0.172822</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.837152</td>
          <td>0.537238</td>
          <td>28.032588</td>
          <td>0.541640</td>
          <td>27.551120</td>
          <td>0.342552</td>
          <td>26.574778</td>
          <td>0.245069</td>
          <td>27.519871</td>
          <td>0.845103</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.564671</td>
          <td>0.445402</td>
          <td>25.800776</td>
          <td>0.088574</td>
          <td>24.800482</td>
          <td>0.032824</td>
          <td>23.822165</td>
          <td>0.022843</td>
          <td>23.093969</td>
          <td>0.022773</td>
          <td>22.858547</td>
          <td>0.041700</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.999169</td>
          <td>0.555771</td>
          <td>28.203143</td>
          <td>0.592055</td>
          <td>26.396588</td>
          <td>0.225708</td>
          <td>26.463045</td>
          <td>0.423804</td>
          <td>25.948480</td>
          <td>0.575085</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.075705</td>
          <td>0.299719</td>
          <td>25.771061</td>
          <td>0.084589</td>
          <td>25.472973</td>
          <td>0.058228</td>
          <td>24.829799</td>
          <td>0.054152</td>
          <td>24.341646</td>
          <td>0.066809</td>
          <td>23.569486</td>
          <td>0.076518</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.927013</td>
          <td>0.580593</td>
          <td>26.303873</td>
          <td>0.137105</td>
          <td>26.265344</td>
          <td>0.119394</td>
          <td>26.031798</td>
          <td>0.158520</td>
          <td>25.881552</td>
          <td>0.256460</td>
          <td>25.061730</td>
          <td>0.279705</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.158534</td>
          <td>0.675639</td>
          <td>27.456780</td>
          <td>0.351433</td>
          <td>26.791933</td>
          <td>0.184544</td>
          <td>26.473368</td>
          <td>0.226228</td>
          <td>26.404277</td>
          <td>0.383395</td>
          <td>25.452979</td>
          <td>0.375849</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.901005</td>
          <td>0.225758</td>
          <td>26.968167</td>
          <td>0.215754</td>
          <td>26.488735</td>
          <td>0.231066</td>
          <td>25.864535</td>
          <td>0.250976</td>
          <td>25.411554</td>
          <td>0.366772</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.775864</td>
          <td>0.206679</td>
          <td>26.765122</td>
          <td>0.185200</td>
          <td>25.930189</td>
          <td>0.146833</td>
          <td>25.403265</td>
          <td>0.173715</td>
          <td>25.254812</td>
          <td>0.329789</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>31.940574</td>
          <td>4.641657</td>
          <td>26.875844</td>
          <td>0.220568</td>
          <td>26.013455</td>
          <td>0.094773</td>
          <td>25.561281</td>
          <td>0.104324</td>
          <td>25.304132</td>
          <td>0.156360</td>
          <td>24.639358</td>
          <td>0.195162</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>27.382835</td>
          <td>0.716944</td>
          <td>26.922922</td>
          <td>0.198550</td>
          <td>25.969390</td>
          <td>0.076754</td>
          <td>25.365660</td>
          <td>0.073424</td>
          <td>25.012403</td>
          <td>0.102566</td>
          <td>24.926470</td>
          <td>0.209683</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.152479</td>
          <td>1.903686</td>
          <td>28.711102</td>
          <td>0.774552</td>
          <td>27.716942</td>
          <td>0.337960</td>
          <td>28.085984</td>
          <td>0.670216</td>
          <td>27.461047</td>
          <td>0.721219</td>
          <td>28.297806</td>
          <td>1.899561</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.344634</td>
          <td>0.730687</td>
          <td>26.000432</td>
          <td>0.096327</td>
          <td>24.790867</td>
          <td>0.029373</td>
          <td>23.876640</td>
          <td>0.021563</td>
          <td>23.115524</td>
          <td>0.020985</td>
          <td>22.822535</td>
          <td>0.036317</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.048362</td>
          <td>1.212277</td>
          <td>27.415109</td>
          <td>0.356894</td>
          <td>27.834307</td>
          <td>0.450622</td>
          <td>26.380627</td>
          <td>0.221959</td>
          <td>25.951195</td>
          <td>0.282352</td>
          <td>25.121636</td>
          <td>0.305552</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.883722</td>
          <td>0.504517</td>
          <td>25.720412</td>
          <td>0.070122</td>
          <td>25.458049</td>
          <td>0.048841</td>
          <td>24.869656</td>
          <td>0.047356</td>
          <td>24.365990</td>
          <td>0.058050</td>
          <td>23.634617</td>
          <td>0.068575</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>27.078118</td>
          <td>0.607074</td>
          <td>26.385531</td>
          <td>0.134269</td>
          <td>25.917727</td>
          <td>0.079366</td>
          <td>25.813107</td>
          <td>0.118075</td>
          <td>26.682172</td>
          <td>0.439774</td>
          <td>24.980817</td>
          <td>0.236947</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.350980</td>
          <td>1.298966</td>
          <td>26.824079</td>
          <td>0.185193</td>
          <td>26.788778</td>
          <td>0.159297</td>
          <td>26.041627</td>
          <td>0.134989</td>
          <td>25.829334</td>
          <td>0.210066</td>
          <td>24.806332</td>
          <td>0.192666</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.829140</td>
          <td>0.498418</td>
          <td>27.622802</td>
          <td>0.364835</td>
          <td>26.858962</td>
          <td>0.174520</td>
          <td>26.125598</td>
          <td>0.150022</td>
          <td>26.098054</td>
          <td>0.270361</td>
          <td>24.765223</td>
          <td>0.192205</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.812656</td>
          <td>0.512480</td>
          <td>27.245178</td>
          <td>0.284754</td>
          <td>26.619817</td>
          <td>0.151683</td>
          <td>25.871809</td>
          <td>0.128927</td>
          <td>25.521543</td>
          <td>0.178076</td>
          <td>25.268389</td>
          <td>0.309901</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.100134</td>
          <td>0.602593</td>
          <td>26.536701</td>
          <td>0.147761</td>
          <td>26.015703</td>
          <td>0.083136</td>
          <td>25.741234</td>
          <td>0.106419</td>
          <td>25.275110</td>
          <td>0.133944</td>
          <td>25.170953</td>
          <td>0.266533</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
