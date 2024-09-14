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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f4368f40700>



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
          <td>27.666725</td>
          <td>0.863269</td>
          <td>26.805277</td>
          <td>0.179775</td>
          <td>25.987387</td>
          <td>0.077973</td>
          <td>25.299417</td>
          <td>0.069234</td>
          <td>24.964749</td>
          <td>0.098360</td>
          <td>25.344971</td>
          <td>0.295699</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.378543</td>
          <td>0.714824</td>
          <td>27.989501</td>
          <td>0.465421</td>
          <td>27.267546</td>
          <td>0.234643</td>
          <td>27.354904</td>
          <td>0.392237</td>
          <td>27.060057</td>
          <td>0.544547</td>
          <td>26.638015</td>
          <td>0.768425</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.093649</td>
          <td>0.272916</td>
          <td>26.136017</td>
          <td>0.100949</td>
          <td>24.805332</td>
          <td>0.027405</td>
          <td>23.891352</td>
          <td>0.020083</td>
          <td>23.131543</td>
          <td>0.019642</td>
          <td>22.815413</td>
          <td>0.033130</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.899038</td>
          <td>0.996717</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.052562</td>
          <td>0.196105</td>
          <td>26.422542</td>
          <td>0.183858</td>
          <td>25.954214</td>
          <td>0.229482</td>
          <td>25.434875</td>
          <td>0.317800</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.300986</td>
          <td>0.322420</td>
          <td>25.772671</td>
          <td>0.073344</td>
          <td>25.461651</td>
          <td>0.048927</td>
          <td>24.849207</td>
          <td>0.046433</td>
          <td>24.320908</td>
          <td>0.055693</td>
          <td>23.636276</td>
          <td>0.068574</td>
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
          <td>26.510597</td>
          <td>0.380142</td>
          <td>26.234895</td>
          <td>0.110054</td>
          <td>26.161675</td>
          <td>0.090917</td>
          <td>25.881398</td>
          <td>0.115485</td>
          <td>25.865695</td>
          <td>0.213186</td>
          <td>26.020727</td>
          <td>0.498775</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.897136</td>
          <td>0.194271</td>
          <td>26.575624</td>
          <td>0.130485</td>
          <td>26.873900</td>
          <td>0.267593</td>
          <td>26.122932</td>
          <td>0.263673</td>
          <td>25.318475</td>
          <td>0.289445</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.268125</td>
          <td>0.663006</td>
          <td>27.278054</td>
          <td>0.266423</td>
          <td>26.917994</td>
          <td>0.175019</td>
          <td>26.282172</td>
          <td>0.163185</td>
          <td>26.618410</td>
          <td>0.391141</td>
          <td>25.380359</td>
          <td>0.304233</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.392078</td>
          <td>0.292239</td>
          <td>26.654557</td>
          <td>0.139691</td>
          <td>25.785974</td>
          <td>0.106260</td>
          <td>25.815048</td>
          <td>0.204340</td>
          <td>25.368092</td>
          <td>0.301251</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.800297</td>
          <td>0.473895</td>
          <td>26.746729</td>
          <td>0.171063</td>
          <td>26.057709</td>
          <td>0.082965</td>
          <td>25.693869</td>
          <td>0.098028</td>
          <td>25.084642</td>
          <td>0.109237</td>
          <td>24.770387</td>
          <td>0.183860</td>
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
          <td>26.533806</td>
          <td>0.428768</td>
          <td>26.736692</td>
          <td>0.194643</td>
          <td>26.187162</td>
          <td>0.109236</td>
          <td>25.271802</td>
          <td>0.080061</td>
          <td>25.054218</td>
          <td>0.124833</td>
          <td>25.354174</td>
          <td>0.346543</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.733252</td>
          <td>0.977502</td>
          <td>29.200522</td>
          <td>1.154457</td>
          <td>26.862404</td>
          <td>0.195103</td>
          <td>27.310156</td>
          <td>0.438930</td>
          <td>28.821374</td>
          <td>1.726660</td>
          <td>27.983600</td>
          <td>1.797119</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.056529</td>
          <td>0.636582</td>
          <td>25.796503</td>
          <td>0.088243</td>
          <td>24.858037</td>
          <td>0.034531</td>
          <td>23.902052</td>
          <td>0.024471</td>
          <td>23.184790</td>
          <td>0.024626</td>
          <td>22.822702</td>
          <td>0.040398</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.931896</td>
          <td>0.599688</td>
          <td>28.505990</td>
          <td>0.787736</td>
          <td>27.131417</td>
          <td>0.259859</td>
          <td>26.504959</td>
          <td>0.246861</td>
          <td>27.193719</td>
          <td>0.717049</td>
          <td>24.875483</td>
          <td>0.251052</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.977711</td>
          <td>1.129131</td>
          <td>25.909337</td>
          <td>0.095505</td>
          <td>25.355717</td>
          <td>0.052476</td>
          <td>24.864145</td>
          <td>0.055828</td>
          <td>24.228392</td>
          <td>0.060431</td>
          <td>23.740241</td>
          <td>0.088942</td>
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
          <td>26.587852</td>
          <td>0.452846</td>
          <td>26.155461</td>
          <td>0.120590</td>
          <td>25.926763</td>
          <td>0.088792</td>
          <td>25.878931</td>
          <td>0.139021</td>
          <td>26.043363</td>
          <td>0.292522</td>
          <td>24.739642</td>
          <td>0.214563</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.756776</td>
          <td>0.507886</td>
          <td>26.808964</td>
          <td>0.207553</td>
          <td>26.801232</td>
          <td>0.186000</td>
          <td>26.134417</td>
          <td>0.170120</td>
          <td>26.062122</td>
          <td>0.292435</td>
          <td>25.487841</td>
          <td>0.386159</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.494370</td>
          <td>0.848092</td>
          <td>27.578766</td>
          <td>0.389176</td>
          <td>26.898180</td>
          <td>0.203488</td>
          <td>26.563995</td>
          <td>0.245883</td>
          <td>26.074594</td>
          <td>0.297728</td>
          <td>25.061885</td>
          <td>0.277571</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.678314</td>
          <td>0.426293</td>
          <td>26.653512</td>
          <td>0.168470</td>
          <td>26.289417</td>
          <td>0.199273</td>
          <td>25.997748</td>
          <td>0.284648</td>
          <td>25.738794</td>
          <td>0.478710</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.650627</td>
          <td>0.471267</td>
          <td>26.593833</td>
          <td>0.174019</td>
          <td>25.975062</td>
          <td>0.091632</td>
          <td>25.722078</td>
          <td>0.120022</td>
          <td>25.122623</td>
          <td>0.133764</td>
          <td>24.895082</td>
          <td>0.241517</td>
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
          <td>27.085262</td>
          <td>0.583345</td>
          <td>26.572221</td>
          <td>0.147384</td>
          <td>26.139909</td>
          <td>0.089205</td>
          <td>25.296496</td>
          <td>0.069065</td>
          <td>25.025797</td>
          <td>0.103775</td>
          <td>24.825946</td>
          <td>0.192714</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.372360</td>
          <td>0.712206</td>
          <td>28.390283</td>
          <td>0.622805</td>
          <td>27.242128</td>
          <td>0.229960</td>
          <td>27.738945</td>
          <td>0.524018</td>
          <td>27.025182</td>
          <td>0.531346</td>
          <td>25.739154</td>
          <td>0.403735</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.978077</td>
          <td>0.262266</td>
          <td>25.904902</td>
          <td>0.088583</td>
          <td>24.817974</td>
          <td>0.030080</td>
          <td>23.905820</td>
          <td>0.022109</td>
          <td>23.158713</td>
          <td>0.021773</td>
          <td>22.861760</td>
          <td>0.037598</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.811919</td>
          <td>1.059114</td>
          <td>27.486301</td>
          <td>0.377292</td>
          <td>27.789795</td>
          <td>0.435713</td>
          <td>26.389155</td>
          <td>0.223539</td>
          <td>26.045281</td>
          <td>0.304605</td>
          <td>25.783852</td>
          <td>0.508851</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.876831</td>
          <td>0.228657</td>
          <td>25.801901</td>
          <td>0.075354</td>
          <td>25.459544</td>
          <td>0.048906</td>
          <td>24.829757</td>
          <td>0.045708</td>
          <td>24.378773</td>
          <td>0.058712</td>
          <td>23.693391</td>
          <td>0.072236</td>
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
          <td>26.505964</td>
          <td>0.397859</td>
          <td>26.292651</td>
          <td>0.123904</td>
          <td>26.143207</td>
          <td>0.096784</td>
          <td>26.077696</td>
          <td>0.148423</td>
          <td>25.725237</td>
          <td>0.204282</td>
          <td>26.277415</td>
          <td>0.640838</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.889099</td>
          <td>0.510979</td>
          <td>26.734409</td>
          <td>0.171643</td>
          <td>27.026835</td>
          <td>0.194956</td>
          <td>26.235395</td>
          <td>0.159450</td>
          <td>25.982626</td>
          <td>0.238610</td>
          <td>25.623967</td>
          <td>0.374591</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.653154</td>
          <td>1.537984</td>
          <td>27.132279</td>
          <td>0.246037</td>
          <td>26.861001</td>
          <td>0.174822</td>
          <td>26.390559</td>
          <td>0.187994</td>
          <td>26.395244</td>
          <td>0.343145</td>
          <td>26.195334</td>
          <td>0.590186</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>30.512647</td>
          <td>3.208003</td>
          <td>27.314127</td>
          <td>0.301031</td>
          <td>26.725168</td>
          <td>0.165980</td>
          <td>26.061519</td>
          <td>0.151826</td>
          <td>26.108540</td>
          <td>0.289705</td>
          <td>25.413349</td>
          <td>0.347710</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.694379</td>
          <td>0.448027</td>
          <td>26.480270</td>
          <td>0.140764</td>
          <td>25.988816</td>
          <td>0.081188</td>
          <td>25.697083</td>
          <td>0.102388</td>
          <td>25.191374</td>
          <td>0.124577</td>
          <td>24.806379</td>
          <td>0.197018</td>
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
