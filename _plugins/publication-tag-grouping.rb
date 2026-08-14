# Group publications by the first tag while preserving all tags in BibTeX.
module PublicationTagGrouping
  def group_value(key, item)
    return super unless key == 'tags'

    item[key].to_s.split(/\s*,\s*/).first.to_s
  end

  def group_compare(key, value_one, value_two)
    return super unless key == 'tags'

    tag_order = Array(config['publication_tag_order'])
    index_one = tag_order.index(value_one)
    index_two = tag_order.index(value_two)

    return value_one <=> value_two if index_one.nil? && index_two.nil?
    return 1 if index_one.nil?
    return -1 if index_two.nil?

    index_one <=> index_two
  end
end

Jekyll::Scholar::Utilities.prepend(PublicationTagGrouping)
