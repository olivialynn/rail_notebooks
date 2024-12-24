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

    <pzflow.flow.Flow at 0x7f225e7f5e40>



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
          <td>27.928424</td>
          <td>1.014456</td>
          <td>26.748134</td>
          <td>0.171267</td>
          <td>26.074364</td>
          <td>0.084192</td>
          <td>25.292227</td>
          <td>0.068795</td>
          <td>25.079588</td>
          <td>0.108756</td>
          <td>24.799821</td>
          <td>0.188490</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.397937</td>
          <td>0.625743</td>
          <td>27.998482</td>
          <td>0.420302</td>
          <td>27.093954</td>
          <td>0.319566</td>
          <td>27.091446</td>
          <td>0.557031</td>
          <td>27.332196</td>
          <td>1.174859</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.034818</td>
          <td>0.260145</td>
          <td>25.884350</td>
          <td>0.080935</td>
          <td>24.745334</td>
          <td>0.026009</td>
          <td>23.886195</td>
          <td>0.019996</td>
          <td>23.137783</td>
          <td>0.019746</td>
          <td>22.815285</td>
          <td>0.033127</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.099918</td>
          <td>1.121728</td>
          <td>28.897987</td>
          <td>0.873331</td>
          <td>27.321869</td>
          <td>0.245403</td>
          <td>26.495767</td>
          <td>0.195576</td>
          <td>25.721173</td>
          <td>0.188824</td>
          <td>25.220495</td>
          <td>0.267316</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.941036</td>
          <td>1.022128</td>
          <td>25.742989</td>
          <td>0.071446</td>
          <td>25.385919</td>
          <td>0.045746</td>
          <td>24.810026</td>
          <td>0.044846</td>
          <td>24.445065</td>
          <td>0.062179</td>
          <td>23.730041</td>
          <td>0.074505</td>
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
          <td>27.197944</td>
          <td>0.631520</td>
          <td>26.414596</td>
          <td>0.128645</td>
          <td>26.138253</td>
          <td>0.089064</td>
          <td>26.199998</td>
          <td>0.152106</td>
          <td>25.987709</td>
          <td>0.235936</td>
          <td>25.203911</td>
          <td>0.263722</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.310494</td>
          <td>1.261904</td>
          <td>27.206805</td>
          <td>0.251334</td>
          <td>26.719369</td>
          <td>0.147705</td>
          <td>26.386019</td>
          <td>0.178258</td>
          <td>26.105919</td>
          <td>0.260031</td>
          <td>25.263609</td>
          <td>0.276863</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.220198</td>
          <td>0.641383</td>
          <td>27.089774</td>
          <td>0.228198</td>
          <td>26.861341</td>
          <td>0.166785</td>
          <td>26.687920</td>
          <td>0.229636</td>
          <td>26.416705</td>
          <td>0.334013</td>
          <td>25.406340</td>
          <td>0.310635</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.702780</td>
          <td>0.440439</td>
          <td>27.658764</td>
          <td>0.361249</td>
          <td>26.494589</td>
          <td>0.121631</td>
          <td>25.875838</td>
          <td>0.114927</td>
          <td>25.739581</td>
          <td>0.191779</td>
          <td>25.120879</td>
          <td>0.246364</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.430242</td>
          <td>1.345573</td>
          <td>26.434956</td>
          <td>0.130931</td>
          <td>26.082440</td>
          <td>0.084793</td>
          <td>25.604764</td>
          <td>0.090650</td>
          <td>25.209537</td>
          <td>0.121787</td>
          <td>25.294254</td>
          <td>0.283830</td>
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
          <td>27.498041</td>
          <td>0.844087</td>
          <td>26.866676</td>
          <td>0.217021</td>
          <td>26.025132</td>
          <td>0.094793</td>
          <td>25.333850</td>
          <td>0.084562</td>
          <td>25.024251</td>
          <td>0.121629</td>
          <td>25.012726</td>
          <td>0.263452</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.504170</td>
          <td>0.751763</td>
          <td>27.339837</td>
          <td>0.289361</td>
          <td>27.280509</td>
          <td>0.429167</td>
          <td>27.419205</td>
          <td>0.791831</td>
          <td>25.865153</td>
          <td>0.511610</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.667880</td>
          <td>0.481184</td>
          <td>25.999211</td>
          <td>0.105382</td>
          <td>24.804890</td>
          <td>0.032951</td>
          <td>23.914265</td>
          <td>0.024731</td>
          <td>23.119448</td>
          <td>0.023277</td>
          <td>22.819876</td>
          <td>0.040297</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.008131</td>
          <td>1.187504</td>
          <td>28.526476</td>
          <td>0.798350</td>
          <td>27.373735</td>
          <td>0.316098</td>
          <td>26.787004</td>
          <td>0.310386</td>
          <td>26.269647</td>
          <td>0.365026</td>
          <td>25.164014</td>
          <td>0.317143</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.027056</td>
          <td>0.288210</td>
          <td>25.780455</td>
          <td>0.085290</td>
          <td>25.480996</td>
          <td>0.058644</td>
          <td>24.952950</td>
          <td>0.060403</td>
          <td>24.400511</td>
          <td>0.070381</td>
          <td>23.728339</td>
          <td>0.088016</td>
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
          <td>26.339721</td>
          <td>0.374557</td>
          <td>26.256831</td>
          <td>0.131652</td>
          <td>26.092987</td>
          <td>0.102732</td>
          <td>26.062066</td>
          <td>0.162672</td>
          <td>25.619799</td>
          <td>0.206453</td>
          <td>25.648559</td>
          <td>0.443316</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.389955</td>
          <td>0.788911</td>
          <td>27.569551</td>
          <td>0.383772</td>
          <td>26.671043</td>
          <td>0.166547</td>
          <td>26.283746</td>
          <td>0.193047</td>
          <td>25.958273</td>
          <td>0.268819</td>
          <td>25.185466</td>
          <td>0.304207</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.056457</td>
          <td>0.256631</td>
          <td>27.122063</td>
          <td>0.245108</td>
          <td>26.335410</td>
          <td>0.203340</td>
          <td>25.702875</td>
          <td>0.219565</td>
          <td>25.046490</td>
          <td>0.274121</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.602595</td>
          <td>0.917295</td>
          <td>27.443476</td>
          <td>0.355522</td>
          <td>26.454514</td>
          <td>0.142076</td>
          <td>25.730901</td>
          <td>0.123618</td>
          <td>25.945893</td>
          <td>0.272919</td>
          <td>25.467724</td>
          <td>0.389672</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.319664</td>
          <td>0.366019</td>
          <td>26.557005</td>
          <td>0.168659</td>
          <td>26.154982</td>
          <td>0.107275</td>
          <td>25.653268</td>
          <td>0.113047</td>
          <td>25.315690</td>
          <td>0.157914</td>
          <td>24.827685</td>
          <td>0.228422</td>
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
          <td>26.696587</td>
          <td>0.438415</td>
          <td>26.923812</td>
          <td>0.198699</td>
          <td>26.096720</td>
          <td>0.085878</td>
          <td>25.251013</td>
          <td>0.066338</td>
          <td>24.982844</td>
          <td>0.099945</td>
          <td>24.951859</td>
          <td>0.214179</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.563374</td>
          <td>0.808228</td>
          <td>28.983197</td>
          <td>0.921766</td>
          <td>27.334472</td>
          <td>0.248182</td>
          <td>26.994696</td>
          <td>0.295394</td>
          <td>26.681196</td>
          <td>0.410847</td>
          <td>28.077993</td>
          <td>1.721309</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.378664</td>
          <td>0.361290</td>
          <td>26.049373</td>
          <td>0.100543</td>
          <td>24.847133</td>
          <td>0.030860</td>
          <td>23.866761</td>
          <td>0.021382</td>
          <td>23.152233</td>
          <td>0.021653</td>
          <td>22.841946</td>
          <td>0.036945</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.074959</td>
          <td>0.661204</td>
          <td>28.680414</td>
          <td>0.879126</td>
          <td>27.772045</td>
          <td>0.429881</td>
          <td>26.840332</td>
          <td>0.322803</td>
          <td>25.822666</td>
          <td>0.254266</td>
          <td>25.741590</td>
          <td>0.493238</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.206147</td>
          <td>0.299151</td>
          <td>25.680741</td>
          <td>0.067706</td>
          <td>25.404589</td>
          <td>0.046577</td>
          <td>24.807566</td>
          <td>0.044816</td>
          <td>24.306592</td>
          <td>0.055069</td>
          <td>23.584869</td>
          <td>0.065619</td>
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
          <td>26.734013</td>
          <td>0.472932</td>
          <td>26.303266</td>
          <td>0.125049</td>
          <td>26.326202</td>
          <td>0.113577</td>
          <td>25.905214</td>
          <td>0.127903</td>
          <td>26.569266</td>
          <td>0.403478</td>
          <td>25.622989</td>
          <td>0.396187</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.370251</td>
          <td>0.717122</td>
          <td>26.738693</td>
          <td>0.172269</td>
          <td>27.061304</td>
          <td>0.200688</td>
          <td>27.307382</td>
          <td>0.383869</td>
          <td>26.710382</td>
          <td>0.425811</td>
          <td>27.903354</td>
          <td>1.598917</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.563550</td>
          <td>0.408143</td>
          <td>27.387915</td>
          <td>0.302878</td>
          <td>26.752774</td>
          <td>0.159423</td>
          <td>26.523806</td>
          <td>0.210267</td>
          <td>26.025834</td>
          <td>0.254865</td>
          <td>25.401879</td>
          <td>0.324120</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.767998</td>
          <td>0.495912</td>
          <td>27.385693</td>
          <td>0.318776</td>
          <td>26.865835</td>
          <td>0.187022</td>
          <td>25.822482</td>
          <td>0.123530</td>
          <td>26.138384</td>
          <td>0.296761</td>
          <td>28.715671</td>
          <td>2.373280</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.519616</td>
          <td>0.145609</td>
          <td>26.292844</td>
          <td>0.106039</td>
          <td>25.608880</td>
          <td>0.094770</td>
          <td>25.029480</td>
          <td>0.108203</td>
          <td>24.801163</td>
          <td>0.196155</td>
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
