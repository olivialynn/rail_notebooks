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

    <pzflow.flow.Flow at 0x7f5f8e033160>



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
          <td>27.091236</td>
          <td>0.585787</td>
          <td>26.730824</td>
          <td>0.168765</td>
          <td>26.242597</td>
          <td>0.097613</td>
          <td>25.350122</td>
          <td>0.072412</td>
          <td>24.989184</td>
          <td>0.100489</td>
          <td>25.222158</td>
          <td>0.267679</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.119736</td>
          <td>0.512589</td>
          <td>28.196598</td>
          <td>0.487884</td>
          <td>27.258383</td>
          <td>0.363883</td>
          <td>26.627659</td>
          <td>0.393946</td>
          <td>25.463940</td>
          <td>0.325243</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.718449</td>
          <td>0.445682</td>
          <td>26.026501</td>
          <td>0.091712</td>
          <td>24.760363</td>
          <td>0.026351</td>
          <td>23.866974</td>
          <td>0.019673</td>
          <td>23.144360</td>
          <td>0.019856</td>
          <td>22.829245</td>
          <td>0.033537</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.653572</td>
          <td>0.856078</td>
          <td>28.915122</td>
          <td>0.882832</td>
          <td>27.112584</td>
          <td>0.206240</td>
          <td>26.394991</td>
          <td>0.179619</td>
          <td>26.281552</td>
          <td>0.299852</td>
          <td>24.724404</td>
          <td>0.176835</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.195979</td>
          <td>0.296449</td>
          <td>25.749283</td>
          <td>0.071845</td>
          <td>25.454311</td>
          <td>0.048610</td>
          <td>24.740334</td>
          <td>0.042157</td>
          <td>24.426904</td>
          <td>0.061185</td>
          <td>23.652572</td>
          <td>0.069570</td>
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
          <td>26.652952</td>
          <td>0.424099</td>
          <td>26.100862</td>
          <td>0.097890</td>
          <td>26.081283</td>
          <td>0.084707</td>
          <td>26.242005</td>
          <td>0.157679</td>
          <td>25.727624</td>
          <td>0.189855</td>
          <td>25.493663</td>
          <td>0.333010</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.774435</td>
          <td>0.464829</td>
          <td>26.784229</td>
          <td>0.176597</td>
          <td>26.532584</td>
          <td>0.125709</td>
          <td>26.450163</td>
          <td>0.188200</td>
          <td>26.034025</td>
          <td>0.245129</td>
          <td>24.935232</td>
          <td>0.211198</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.723895</td>
          <td>1.561987</td>
          <td>28.348662</td>
          <td>0.604442</td>
          <td>26.940639</td>
          <td>0.178415</td>
          <td>26.588270</td>
          <td>0.211354</td>
          <td>26.193745</td>
          <td>0.279321</td>
          <td>25.130539</td>
          <td>0.248329</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.112333</td>
          <td>0.594625</td>
          <td>27.177666</td>
          <td>0.245385</td>
          <td>26.769685</td>
          <td>0.154221</td>
          <td>25.914230</td>
          <td>0.118832</td>
          <td>25.741817</td>
          <td>0.192140</td>
          <td>25.305269</td>
          <td>0.286372</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.852656</td>
          <td>0.969107</td>
          <td>26.846908</td>
          <td>0.186216</td>
          <td>26.094756</td>
          <td>0.085718</td>
          <td>25.639591</td>
          <td>0.093467</td>
          <td>25.208012</td>
          <td>0.121626</td>
          <td>24.914631</td>
          <td>0.207590</td>
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
          <td>27.010675</td>
          <td>0.608144</td>
          <td>26.891885</td>
          <td>0.221622</td>
          <td>25.949779</td>
          <td>0.088721</td>
          <td>25.277529</td>
          <td>0.080466</td>
          <td>24.875082</td>
          <td>0.106811</td>
          <td>24.961553</td>
          <td>0.252641</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.446818</td>
          <td>0.347575</td>
          <td>28.510375</td>
          <td>0.695218</td>
          <td>26.763286</td>
          <td>0.285840</td>
          <td>26.636932</td>
          <td>0.456371</td>
          <td>26.653986</td>
          <td>0.877783</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.694342</td>
          <td>0.490719</td>
          <td>26.029090</td>
          <td>0.108165</td>
          <td>24.786082</td>
          <td>0.032410</td>
          <td>23.927445</td>
          <td>0.025015</td>
          <td>23.174825</td>
          <td>0.024415</td>
          <td>22.742718</td>
          <td>0.037639</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.448176</td>
          <td>0.848984</td>
          <td>27.787948</td>
          <td>0.476064</td>
          <td>27.381278</td>
          <td>0.318006</td>
          <td>26.628947</td>
          <td>0.273218</td>
          <td>25.988054</td>
          <td>0.291850</td>
          <td>25.836957</td>
          <td>0.530623</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.051095</td>
          <td>0.625768</td>
          <td>25.706545</td>
          <td>0.079920</td>
          <td>25.384234</td>
          <td>0.053820</td>
          <td>24.835681</td>
          <td>0.054436</td>
          <td>24.433192</td>
          <td>0.072444</td>
          <td>23.578155</td>
          <td>0.077105</td>
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
          <td>26.377858</td>
          <td>0.385801</td>
          <td>26.417474</td>
          <td>0.151167</td>
          <td>26.325442</td>
          <td>0.125788</td>
          <td>26.150865</td>
          <td>0.175444</td>
          <td>25.238544</td>
          <td>0.149402</td>
          <td>28.108634</td>
          <td>1.919442</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.133562</td>
          <td>0.664151</td>
          <td>27.458671</td>
          <td>0.351956</td>
          <td>27.055119</td>
          <td>0.230049</td>
          <td>26.540130</td>
          <td>0.239085</td>
          <td>27.179506</td>
          <td>0.676452</td>
          <td>25.566499</td>
          <td>0.410288</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.487293</td>
          <td>0.417355</td>
          <td>26.967816</td>
          <td>0.238593</td>
          <td>27.224321</td>
          <td>0.266538</td>
          <td>26.581958</td>
          <td>0.249543</td>
          <td>26.843971</td>
          <td>0.537473</td>
          <td>25.174597</td>
          <td>0.304011</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.005317</td>
          <td>0.249994</td>
          <td>26.365292</td>
          <td>0.131548</td>
          <td>25.914600</td>
          <td>0.144878</td>
          <td>25.677222</td>
          <td>0.218749</td>
          <td>25.139504</td>
          <td>0.300773</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.618386</td>
          <td>0.177680</td>
          <td>26.155296</td>
          <td>0.107305</td>
          <td>25.614852</td>
          <td>0.109322</td>
          <td>25.459056</td>
          <td>0.178417</td>
          <td>24.467785</td>
          <td>0.168781</td>
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
          <td>27.057988</td>
          <td>0.572102</td>
          <td>26.738608</td>
          <td>0.169904</td>
          <td>26.054577</td>
          <td>0.082747</td>
          <td>25.316360</td>
          <td>0.070290</td>
          <td>24.980962</td>
          <td>0.099780</td>
          <td>25.492183</td>
          <td>0.332661</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.126563</td>
          <td>0.600960</td>
          <td>27.980757</td>
          <td>0.462706</td>
          <td>27.563179</td>
          <td>0.298944</td>
          <td>27.456645</td>
          <td>0.424454</td>
          <td>26.399886</td>
          <td>0.329869</td>
          <td>25.809887</td>
          <td>0.426186</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.689973</td>
          <td>0.913413</td>
          <td>25.873482</td>
          <td>0.086170</td>
          <td>24.757963</td>
          <td>0.028539</td>
          <td>23.883278</td>
          <td>0.021686</td>
          <td>23.106268</td>
          <td>0.020820</td>
          <td>22.776189</td>
          <td>0.034861</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.154777</td>
          <td>1.917588</td>
          <td>27.074695</td>
          <td>0.247218</td>
          <td>26.347503</td>
          <td>0.215919</td>
          <td>25.749699</td>
          <td>0.239450</td>
          <td>25.073267</td>
          <td>0.293897</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.845055</td>
          <td>0.222712</td>
          <td>25.919010</td>
          <td>0.083547</td>
          <td>25.442587</td>
          <td>0.048175</td>
          <td>24.881268</td>
          <td>0.047846</td>
          <td>24.336991</td>
          <td>0.056575</td>
          <td>23.792218</td>
          <td>0.078828</td>
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
          <td>26.149237</td>
          <td>0.300494</td>
          <td>26.432064</td>
          <td>0.139765</td>
          <td>26.247651</td>
          <td>0.106052</td>
          <td>26.129184</td>
          <td>0.155124</td>
          <td>25.915136</td>
          <td>0.239251</td>
          <td>25.155963</td>
          <td>0.273542</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.633107</td>
          <td>2.325686</td>
          <td>26.887560</td>
          <td>0.195373</td>
          <td>26.972193</td>
          <td>0.186175</td>
          <td>27.087612</td>
          <td>0.322975</td>
          <td>26.706530</td>
          <td>0.424563</td>
          <td>25.194328</td>
          <td>0.265859</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.698491</td>
          <td>0.452189</td>
          <td>27.244831</td>
          <td>0.269784</td>
          <td>26.922982</td>
          <td>0.184250</td>
          <td>26.312096</td>
          <td>0.175912</td>
          <td>26.408739</td>
          <td>0.346816</td>
          <td>25.289454</td>
          <td>0.296223</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.951533</td>
          <td>1.814963</td>
          <td>26.900064</td>
          <td>0.214418</td>
          <td>26.544154</td>
          <td>0.142135</td>
          <td>26.265365</td>
          <td>0.180637</td>
          <td>25.510200</td>
          <td>0.176371</td>
          <td>25.865332</td>
          <td>0.491295</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.348851</td>
          <td>0.343206</td>
          <td>26.613707</td>
          <td>0.157836</td>
          <td>26.115710</td>
          <td>0.090787</td>
          <td>25.613376</td>
          <td>0.095144</td>
          <td>25.110544</td>
          <td>0.116127</td>
          <td>24.446961</td>
          <td>0.145104</td>
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
